#!/usr/bin/env python3
"""
Консольная программа: генерация карт osu!mania 4K нейросетью — по шагам, без запоминания ключей.

    python mapper_console.py            (на Windows: двойной клик по composer.bat — имя файла осталось старым)

Программа спрашивает песню, сложности, лонгноты и т.д. простыми вопросами, показывает итог, запускает генерацию
(generate_v5.py, если есть checkpoints/v5/v5_best.pt, иначе generate_hybrid.py) и запоминает выбор до следующего раза. В конце печатает ту же генерацию одной командой - её
можно сохранить или запустить без вопросов.

Композитор (generate_composer.py) устарел (deprecated) и из консоли убран: он ставил много лишних нот, но не
уплотнял те места, где это действительно нужно.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent
SETTINGS = Path.home() / ".aimaniamapper" / "composer_console.json"
AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus")

# ----------------------------------------------------------------------------------------------
# small console helpers
# ----------------------------------------------------------------------------------------------

def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows консоль: кириллица и графики плотности
        except Exception:
            pass
    if os.name == "nt":
        os.system("")  # включает цвета ANSI в старых консолях Windows


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def title(text: str) -> None:
    print()
    print(color(f"── {text} " + "─" * max(0, 60 - len(text)), "1;36"))


def ask(prompt: str, default: Optional[str] = None) -> str:
    hint = f" [{default}]" if default not in (None, "") else ""
    try:
        value = input(f"{prompt}{hint}: ").strip()
    except EOFError:
        print()
        raise SystemExit(0)
    return value if value else (default or "")


def ask_yes(prompt: str, default: bool = True) -> bool:
    while True:
        v = ask(prompt + (" (Д/н)" if default else " (д/Н)")).lower()
        if not v:
            return default
        if v in ("д", "да", "y", "yes", "1", "l"):
            return True
        if v in ("н", "нет", "n", "no", "0", "т"):
            return False
        print("  Ответьте «д» или «н».")


def ask_choice(question: str, options: Sequence[Tuple[str, str]], default: int = 1) -> int:
    """Нумерованное меню; возвращает номер (с 1)."""
    print(question)
    for i, (name, hint) in enumerate(options, 1):
        mark = color("●", "32") if i == default else " "
        print(f"  {mark} {i}. {name}" + (color(f" — {hint}", "2") if hint else ""))
    while True:
        v = ask("Номер", str(default))
        if v.isdigit() and 1 <= int(v) <= len(options):
            return int(v)
        print(f"  Введите число от 1 до {len(options)}.")


def clean_path(text: str) -> str:
    """Путь, перетащенный в консоль: убирает кавычки и экранирование пробелов."""
    t = text.strip().strip("&").strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1]
    if os.name != "nt":
        t = t.replace("\\ ", " ")
    return t


def parse_stars(text: str) -> List[float]:
    """«4», «2 4 6», «2.5 3.5», «2,5 3,5» (запятая в числе — десятичная), «2,4,6» (несколько запятых — список),
    «2-8» (каждую звезду от 2 до 8). Результат отсортирован, без повторов."""
    out: List[float] = []
    for part in text.replace(";", " ").split():
        if "-" in part[1:]:
            a, b = (float(x.replace(",", ".")) for x in part.split("-", 1))
            out += [round(a + k, 2) for k in range(int(b - a + 1e-9) + 1)]
        elif part.count(",") == 1:
            out.append(round(float(part.replace(",", ".")), 2))
        else:
            out += [round(float(x), 2) for x in part.split(",") if x]
    return sorted(set(out))


# ----------------------------------------------------------------------------------------------
# reading the song
# ----------------------------------------------------------------------------------------------

def read_osz(path: Path) -> Dict[str, object]:
    """Название, BPM и 4K-сложности шаблона (.osz / .osu)."""
    sys.path.insert(0, str(REPO))
    from osu_mapper.hybrid.beatmap import parse_osu_mania, _kv, _split_sections

    texts: List[Tuple[str, str]] = []
    if path.suffix.lower() == ".osz":
        with zipfile.ZipFile(path) as z:
            texts = [(n, z.read(n).decode("utf-8", errors="replace")) for n in z.namelist() if n.lower().endswith(".osu")]
    else:
        texts = [(path.name, path.read_text(encoding="utf-8", errors="replace"))]
    info: Dict[str, object] = {"title": "", "artist": "", "bpm": None, "diffs": []}
    for name, text in texts:
        meta = _kv(_split_sections(text).get("Metadata", []))
        info["title"] = info["title"] or meta.get("Title", "")
        info["artist"] = info["artist"] or meta.get("Artist", "")
        m = parse_osu_mania(text)
        if m is None:
            continue
        reds = [t for t in m.timing_points if t.uninherited and t.beat_len > 0]
        if reds and info["bpm"] is None:
            info["bpm"] = round(60000.0 / reds[0].beat_len, 2)
        if m.notes:
            try:
                from osu_mapper.hybrid.composer import star_rating
                sr = star_rating(m.notes, m.timing_points, m.overall_difficulty)
            except Exception:
                sr = 0.0
            info["diffs"].append((m.metadata.get("Version", name), round(sr, 2), len(m.notes)))
    info["diffs"] = sorted(info["diffs"], key=lambda d: d[1])
    return info


# ----------------------------------------------------------------------------------------------
# the dialog
# ----------------------------------------------------------------------------------------------

def load_settings() -> Dict[str, object]:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(s: Dict[str, object]) -> None:
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


MODEL_DIRS = (REPO / "checkpoints" / "hybrid", REPO / "test_NN")


def find_model(name: str) -> Optional[Path]:
    """checkpoints/hybrid/<name>, else test_NN/<name> (weights committed to the repository)."""
    for d in MODEL_DIRS:
        p = d / name
        if p.is_file() and p.stat().st_size > 1024:  # not a Git LFS pointer
            return p
    return None


V5_PATHS = (REPO / "checkpoints" / "v5" / "v5_best.pt", REPO / "checkpoints" / "hybrid" / "v5_best.pt",
            REPO / "test_NN" / "v5_best.pt")


def find_v5() -> Optional[Path]:
    """Сеть v5 (train_v5.py): checkpoints/v5/v5_best.pt, иначе checkpoints/hybrid/ или test_NN/."""
    for p in V5_PATHS:
        if p.is_file() and p.stat().st_size > 1024:
            return p
    return None


def model_argv() -> List[str]:
    out: List[str] = []
    ck, pt = find_model("hybrid_best.pt"), find_model("patterns.npz")
    if ck is not None:
        out += ["--checkpoint", str(ck)]
    if pt is not None:
        out += ["--patterns", str(pt)]
    return out


def status_line() -> Tuple[bool, bool]:
    patterns = find_model("patterns.npz") is not None
    net = find_model("hybrid_best.pt") is not None
    ok, no = color("есть", "32"), color("нет", "33")
    v5 = find_v5()
    if v5 is not None:
        print(f"Нейросеть v5:     {ok}" + color(f"  ({v5.relative_to(REPO) if v5.is_relative_to(REPO) else v5})", "2"))
    print(f"Модель паттернов: {ok if patterns else no}" + ("" if patterns else color("  (python learn_patterns.py --data dataset_hybrid)", "2")))
    print(f"Нейросеть:        {ok if net else no}" + ("" if net else color("  (нужна: обучите её, python train_console.py)", "2")))
    return patterns, net


def choose_song(last: Dict[str, object]) -> Tuple[List[str], Dict[str, object], Optional[Path]]:
    """argv-часть с песней, сведения о шаблоне, путь к шаблону (для сравнения с картой)."""
    title("1. Песня")
    print("Перетащите сюда файл .osz (лучше всего: из него берётся точный тайминг) или аудио (.mp3/.ogg/.wav).")
    while True:
        p = Path(clean_path(ask("Файл", str(last.get("song", "")) or None)))
        if p.is_file():
            break
        print(color(f"  Файл не найден: {p}", "31"))
    info: Dict[str, object] = {"title": p.stem, "diffs": [], "bpm": None}
    if p.suffix.lower() in (".osz", ".osu"):
        try:
            info = read_osz(p)
        except Exception as err:
            print(color(f"  Не удалось прочитать шаблон: {err}", "31"))
        name = f"{info.get('artist') or ''} - {info.get('title') or p.stem}".strip(" -")
        bpm = f", {info['bpm']:g} BPM" if info.get("bpm") else ""
        print(f"  {color(name, '1')}{bpm}")
        if info["diffs"]:
            print("  4K-сложности в архиве: " + ", ".join(f"{v} ({sr:.2f}*)" for v, sr, _ in info["diffs"]))
        return ["--template", str(p)], info, p
    argv = [str(p)]
    print("Для точного тайминга нужны BPM и офсет первой доли (как в редакторе). Enter — определить автоматически.")
    bpm = ask("BPM", "")
    if bpm:
        off = ask("Офсет, мс", "0")
        argv += ["--bpm", bpm.replace(",", "."), "--offset", off.replace(",", ".")]
    else:
        print(color("  Тайминг будет оценён по звуку — для готовой карты лучше указать точный.", "2"))
    return argv, info, None


def choose_stars(last: Dict[str, object], info: Dict[str, object]) -> List[float]:
    title("2. Сложности (звёзды)")
    print("Одна или несколько через пробел: «4», «2 4 6», «2,5 3,5»; диапазон «2-8» — каждую звезду от 2 до 8.")
    if info.get("diffs"):
        print("  Для сравнения, в архиве: " + ", ".join(f"{sr:.2f}*" for _, sr, _ in info["diffs"]))
    while True:
        try:
            stars = parse_stars(ask("Звёзды", str(last.get("stars", "4"))))
        except ValueError:
            stars = []
        if stars and all(0.5 <= s <= 12 for s in stars):
            print("  Понял: " + ", ".join(f"{s:g}*" for s in stars))
            break
        print("  Например: 3 или 2 4 6 или 2-8 (от 0.5 до 12).")
    if any(s > 9 for s in stars):
        print(color("  10*+ карт в датасетах мало — паттерны таких сложностей опираются на соседние.", "33"))
    return stars


def network_dialog(last, use_last, argv, info, template, stars, song_argv):
    """Questions for the network-first generator (generate_hybrid.py --fit_sr on)."""
    title("4. Лонгноты")
    no_ln = ask_yes("Убрать лонгноты (только тапы)?", bool(last.get("net_no_ln", False)))
    if no_ln:
        argv.append("--no_ln")
    title("5. Дополнительно")
    if template is not None and info.get("diffs"):
        opts = [("Не сравнивать", "")] + [(f"{v}", f"{sr:.2f}*, {n} нот") for v, sr, n in info["diffs"]]
        r = ask_choice("Сравнить результат с картой из архива?", opts, 1)
        if r > 1:
            argv += ["--reference", str(template), "--reference_diff", str(info["diffs"][r - 2][0])]
    hs = last.get("hitsounds", False) if use_last else ask_yes("Добавить хитсаунды?", False)
    if hs:
        argv.append("--hitsounds")
    creator = ask("Автор (Creator)", str(last.get("creator", "AIManiaMapper")))
    argv += ["--creator", creator]
    safe = "".join(c for c in str(info.get("title") or "map") if c.isalnum() or c in " -_()").strip() or "map"
    out = clean_path(ask("Куда сохранить", str(REPO / "output_beatmaps" / f"{safe} (network).osz")))
    argv += ["--output", out]
    title("Итог")
    print(f"  Песня:      {info.get('title') or song_argv[-1]}")
    print(f"  Сложности:  {', '.join(f'{s:g}*' for s in stars)}")
    print(f"  Строит:     нейросеть (звёзды подгоняются, тайминг выравнивается по красной линии)")
    print(f"  Лонгноты:   {'нет' if no_ln else 'как решит сеть'}    Хитсаунды: {'да' if hs else 'нет'}")
    print(f"  Файл:       {out}")
    if not ask_yes("Начать генерацию?", True):
        return None
    last.update({"song": song_argv[1] if song_argv[0] == "--template" else song_argv[0], "engine": "network",
                 "stars": " ".join(f"{s:g}" for s in stars), "net_no_ln": no_ln, "hitsounds": hs, "creator": creator})
    save_settings(last)
    return ("network", argv)


def v5_dialog(last, argv, info, stars, song_argv):
    """Questions for the v5 network (generate_v5.py)."""
    title("4. Лонгноты")
    opts = [("Как решит сеть", "доля LN оценивается по музыке"), ("Только тапы", "без лонгнотов"),
            ("Задать долю LN", "например 0.3 = 30 % нот")]
    r = ask_choice("Лонгноты:", opts, int(last.get("v5_ln_choice", 1)))
    ln_text = "как решит сеть"
    if r == 2:
        argv += ["--style", "ln_ratio=0"]
        ln_text = "нет"
    elif r == 3:
        while True:
            v = ask("Доля LN (0-1)", str(last.get("v5_ln_ratio", "0.3"))).replace(",", ".")
            try:
                if 0 <= float(v) <= 1:
                    break
            except ValueError:
                pass
            print("  Число от 0 до 1, например 0.3.")
        argv += ["--style", f"ln_ratio={float(v):g}"]
        ln_text = f"{100 * float(v):.0f} % нот"
        last["v5_ln_ratio"] = v
    title("5. Дополнительно")
    creator = ask("Автор (Creator)", str(last.get("creator", "AIManiaMapper")))
    argv += ["--creator", creator]
    safe = "".join(c for c in str(info.get("title") or "map") if c.isalnum() or c in " -_()").strip() or "map"
    out = clean_path(ask("Куда сохранить", str(REPO / "output_beatmaps" / f"{safe} (v5).osz")))
    argv += ["--output", out]
    title("Итог")
    print(f"  Песня:      {info.get('title') or song_argv[-1]}")
    print(f"  Сложности:  {', '.join(f'{s:g}*' for s in stars)}")
    print(f"  Строит:     нейросеть v5")
    print(f"  Лонгноты:   {ln_text}")
    print(f"  Файл:       {out}")
    if not ask_yes("Начать генерацию?", True):
        return None
    last.update({"song": song_argv[1] if song_argv[0] == "--template" else song_argv[0], "engine": "v5",
                 "stars": " ".join(f"{s:g}" for s in stars), "v5_ln_choice": r, "creator": creator})
    save_settings(last)
    return ("v5", argv)


def choose_engine(last, v5: Optional[Path], net: bool) -> str:
    if v5 is None:
        return "network"
    if not net:
        return "v5"
    title("0. Нейросеть")
    opts = [("v5", f"{v5.name} — новая (train_v5.py)"), ("hybrid", "hybrid_best.pt — прежняя (train_hybrid.py)")]
    r = ask_choice("Какой сетью строить?", opts, 2 if last.get("engine") == "network" else 1)
    return "v5" if r == 1 else "network"


def run_dialog() -> Optional[List[str]]:
    last = load_settings()
    patterns, net = status_line()
    v5 = find_v5()
    engine = choose_engine(last, v5, net)
    if engine == "v5":
        song_argv, info, template = choose_song(last)
        stars = choose_stars(last, info)
        argv = song_argv + ["--sr", ",".join(f"{s:g}" for s in stars), "--checkpoint", str(v5)]
        return v5_dialog(last, argv, info, stars, song_argv)
    if not net:
        print(color("Нет обученной нейросети (hybrid_best.pt): карты строит только она. Обучите её: python train_console.py", "31"))
        print(color("(Композитор устарел и из консоли убран: python generate_composer.py, если он всё же нужен.)", "2"))
        return None
    use_last = bool(last) and ask_yes("Взять настройки прошлого раза (песню и сложности спрошу заново)?", True)
    song_argv, info, template = choose_song(last)
    stars = choose_stars(last, info)
    argv: List[str] = song_argv + ["--sr", ",".join(f"{s:g}" for s in stars)] + model_argv()

    return network_dialog(last, use_last, argv, info, template, stars, song_argv)


def main() -> None:
    _setup_console()
    os.chdir(REPO)  # модели и output_beatmaps лежат рядом с программой
    sys.path.insert(0, str(REPO))
    print(color("AIManiaMapper — генератор карт osu!mania 4K", "1;35"))
    print(color("Enter = вариант в [скобках]. Ctrl+C — выход.", "2"))
    while True:
        try:
            res = run_dialog()
            if res:
                engine, argv = res
                title("Генерация")
                if engine == "v5":
                    import generate_v5 as gen
                    script = "generate_v5.py"
                else:
                    import generate_hybrid as gen
                    script = "generate_hybrid.py"
                gen.main(argv)
                print()
                print(color("Готово!", "1;32") + " Та же генерация одной командой:")
                print(f"  python {script} " + " ".join(shlex.quote(a) if os.name != "nt" else (f'"{a}"' if " " in a else a) for a in argv))
        except KeyboardInterrupt:
            print("\nВыход.")
            return
        except SystemExit as err:  # ошибки generate_hybrid (нет файла и т.п.)
            if err.code not in (None, 0):
                print(color(f"Ошибка: {err.code}", "31"))
        except Exception:
            print(color("Что-то пошло не так:", "31"))
            traceback.print_exc()
        if not ask_yes("\nСделать ещё одну карту?", True):
            return


if __name__ == "__main__":
    main()
