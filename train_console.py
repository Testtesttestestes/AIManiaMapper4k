#!/usr/bin/env python3
"""
Консольная программа для обучения: чистка датасета, подготовка, обучение сети, диагностика - по шагам.

    python train_console.py            (на Windows: двойной клик по train_console.bat)

Как mapper_console.py: простые вопросы с вариантами по умолчанию в [скобках], итог перед запуском, та же команда в
конце (её можно сохранить и запускать без вопросов), выбор запоминается до следующего раза. Несколько видеокарт
находятся сами, обучение на них запускается через torchrun.

Обучение идёт только по настоящим картам датасета (--real_only): облегчённые (make_derived_maps.py) и
сгенерированные сетью (make_self_maps.py) карты и DPO-пары портили веса, поэтому консоль их больше не делает и не
использует. Перекос датасета в сторону лёгких карт выравнивается выборкой: сложные карты берутся чаще
(--sr_balance и --hard_tilt), так что паттерны сложных мест сеть учит по самим сложным картам.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mapper_console import _setup_console, ask, ask_choice, ask_yes, clean_path, color, title

REPO = Path(__file__).resolve().parent
SETTINGS = Path.home() / ".aimaniamapper" / "train_console.json"

PRESETS = [  # (name, hint, preset, batch per GPU for ~8 GB)
    ("Маленькая (small)", "быстро, для проверки и слабых видеокарт", "small", 24),
    ("Обычная (base)", "рекомендуется: хорошее качество, помещается в 8 ГБ", "base", 12),
    ("Большая (large)", "лучше на большом датасете, медленнее, нужно больше памяти", "large", 6),
    ("Крошечная (tiny)", "только чтобы проверить, что всё запускается", "tiny", 32),
]
LENGTHS = [  # (name, hint, epochs, steps per epoch)
    ("Обычно", "40 эпох x 1000 шагов", 40, 1000),
    ("Быстро", "10 эпох x 500 шагов: посмотреть, что получается", 10, 500),
    ("Долго", "80 эпох x 1000 шагов: большой датасет", 80, 1000),
    ("Задать самому…", "", 0, 0),
]
# training techniques: (key, question, hint, flags when off)
TECHNIQUES = [
    ("rate", "Ускорение карт для сложных звёзд", "x1.1-x1.3 вместе с музыкой, с пометкой «ускорено»", ["--rate_aug", "0"]),
    ("rare", "Больший вес редких сложностей в потерях", "кроп редкой (сложной) карты весит больше", ["--sr_rare", "0"]),
    ("patterns", "Знания паттернов и библиотеки мотивов", "структура карты и как реальные карты продолжают аккорды",
     ["--pattern_weight", "0", "--prior_weight", "0", "--plan_prob", "0"]),
    ("plan", "План по тактам («сначала подумать»)", "техника каждого такта как цель и как вход", ["--plan_prob", "0"]),
    ("fix", "«Сломай и почини» (самокоррекция)", "сеть учится находить и исправлять ошибки", ["--corrupt_prob", "0"]),
    ("canvas", "Холст карты (дорисовка с оглядкой на уже поставленное)", "", ["--canvas_prob", "0", "--corrupt_prob", "0"]),
    ("ladder", "Лестница сложностей", "какие удары остаются на всех сложностях, какие только на сложных", ["--ladder_weight", "0"]),
]
# how much more often hard maps are drawn: (name, hint, sr_balance, hard_tilt)
SAMPLING = [
    ("Сложные заметно чаще", "рекомендуется: звёздные корзины почти выровнены, плюс наклон к сложным", 0.75, 1.5),
    ("Сложные немного чаще", "мягкое выравнивание", 0.5, 0.75),
    ("Сложные намного чаще", "все звёзды поровну и сильный наклон к сложным (риск переобучиться на редких картах)", 1.0, 2.5),
    ("Как в датасете", "по длине карт, без выравнивания", 0.0, 0.0),
]


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


def gpus() -> List[Tuple[str, float]]:
    try:
        import torch

        return [(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2 ** 30)
                for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def ask_int(prompt: str, default: int, lo: int = 1, hi: int = 10 ** 9) -> int:
    while True:
        v = ask(prompt, str(default))
        if v.lstrip("-").isdigit() and lo <= int(v) <= hi:
            return int(v)
        print(f"  Введите целое число от {lo} до {hi}.")


def ask_float(prompt: str, default: float) -> float:
    while True:
        v = ask(prompt, f"{default:g}").replace(",", ".")
        try:
            return float(v)
        except ValueError:
            print("  Введите число.")


def ask_folder(prompt: str, default: str, must_exist: bool = True) -> str:
    while True:
        p = clean_path(ask(prompt, default))
        if not must_exist or Path(p).exists():
            return p
        print(color(f"  Не найдено: {p}", "31"))


def command_text(argv: List[str]) -> str:
    q = (lambda a: shlex.quote(a)) if os.name != "nt" else (lambda a: f'"{a}"' if " " in a else a)
    return " ".join(q(a) for a in argv)


def run(argv: List[str]) -> int:
    """Runs a step as its own process (the same Python), output straight to the console."""
    title("Запуск")
    print(color(command_text(argv), "2"))
    proc = subprocess.Popen(argv, cwd=str(REPO))
    interrupted = False
    while True:                       # Ctrl+C goes to the step too; wait until it has saved and quit
        try:
            code = proc.wait()
            break
        except KeyboardInterrupt:
            interrupted = True
    return INTERRUPTED if interrupted else code


INTERRUPTED = -1000


# ----------------------------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------------------------

def step_clean(last: Dict[str, object]) -> Optional[List[str]]:
    title("Чистка архивов")
    print("В архивах останутся только 4K-сложности и их аудио; видео, фоны, хитсаунды, сториборды, другие режимы и клоны")
    print("удаляются. Сначала будет отчёт, потом вопрос.")
    folder = ask_folder("Папка с .osz", str(last.get("beatmaps", "sample beatmaps")))
    last["beatmaps"] = folder
    if run([sys.executable, "audit_dataset_compatibility.py", "--dir", folder, "--clean", "--skip_audit"]) != 0:
        return None
    if not ask_yes("Удалить всё это сейчас? (безвозвратно)", False):
        return None
    return [sys.executable, "audit_dataset_compatibility.py", "--dir", folder, "--clean", "--apply", "--skip_audit"]


def step_prepare(last: Dict[str, object]) -> Optional[List[str]]:
    title("Подготовка датасета")
    print("Аудио всех карт превращается в признаки для сети, карты — в параметры. Долго: от минут до часов.")
    folder = ask_folder("Папка с .osz", str(last.get("beatmaps", "sample beatmaps")))
    out = ask_folder("Куда положить датасет", str(last.get("data", "dataset_hybrid")), must_exist=False)
    last.update(beatmaps=folder, data=out)
    print(color("  Облегчённые (прореженные) версии карт больше не добавляются: обучение идёт только по настоящим картам.", "2"))
    argv = [sys.executable, "prepare_hybrid_dataset.py", "--input_dir", folder, "--output_dir", out]
    ext = ask_yes("Нейросетевые признаки звука: demucs (вокал / барабаны / бас / остальное отдельно) и MERT"
                  + color(" (рекомендуется: без них сеть не слышит, где поёт вокал; v2 обучалась с ними; дольше, нужен "
                          "pip install demucs transformers)", "2"), bool(last.get("external", True)))
    last["external"] = ext
    if ext:
        argv += ["--external", "demucs_stems", "mert95m"]
    return argv


def dataset_has_stems(data: str) -> Optional[bool]:
    try:
        d = json.loads((Path(data) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return "demucs_stems" in (d.get("feature_config", {}).get("external") or [])


def warn_no_stems(data: str) -> None:
    if dataset_has_stems(data) is False:
        print(color("  [!] В этом датасете нет дорожек demucs и MERT (подготовлен без --external): сеть слышит только "
                    "общий микс и не отличит вокал от барабанов. Пересоберите датасет (пункт «Подготовка датасета», "
                    "ответ «да» на вопрос про нейросетевые признаки).", "33"))


V2_RECIPE = ["--real_only", "--preset", "base", "--canvas_prob", "0", "--corrupt_prob", "0", "--pair_prob", "0",
             "--pair_weight", "0", "--ladder_weight", "0", "--jack_weight", "0", "--error_weight", "0",
             "--pattern_weight", "0", "--prior_weight", "0", "--plan_weight", "0", "--plan_prob", "0", "--rate_aug", "0",
             "--sr_balance", "0", "--sr_rare", "0", "--pitch_shift", "0", "--density_weight", "0"]


def step_v2_recipe(last: Dict[str, object]) -> Optional[List[str]]:
    title("Рецепт v2 (контроль)")
    print("Та же сеть и обучение, что у v2 из test_NN: все добавки v3/v4 выключены (холст, порча, пары, лесенка, паттерны,")
    print("план, ускорение, баланс звёзд). Сравните с обычным обучением на том же датасете: так видно, что дали добавки.")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")), must_exist=False)
    warn_no_stems(data)
    ckdir = ask_folder("Папка для чекпойнтов", str(last.get("ckdir_v2", "checkpoints/v2_recipe")), must_exist=False)
    last.update(data=data, ckdir_v2=ckdir)
    e = ask_int("Эпох", int(last.get("epochs", 40)))
    tail = ["--data", data, "--checkpoint_dir", ckdir, "--epochs", str(e), "--batch_size", "auto", "--patience", "5"] + V2_RECIPE
    return launcher(last, "train_hybrid.py") + tail


def launcher(last: Dict[str, object], script: str) -> List[str]:
    found = gpus()
    n_gpu = ask_int("Сколько видеокарт использовать", int(last.get("gpus", max(1, len(found)))), 1, max(1, len(found))) if len(found) > 1 else 1
    last["gpus"] = n_gpu
    if n_gpu > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}", script]
    return [sys.executable, script]


V5_PRESETS = [("Маленькая", "быстро, для проверки", "tiny"), ("Обычная", "рекомендуется", "base"),
              ("Большая", "больше памяти и данных", "large")]


def step_v5(last: Dict[str, object], resume: bool = False) -> Optional[List[str]]:
    title("Дообучение v5" if resume else "Обучение v5")
    print("v5: сеть ставит ноты по очереди, как пишут текст (авторегрессия), а не решает за каждый кадр отдельно по порогу;")
    print("слышит дорожки отдельно и учит, за каким звуком идёт каждая нота; лучший чекпойнт выбирается по F1 и")
    print("выразительности (рис/вспышки, ритм, плотность по тактам). Подробности: HYBRID_GUIDE.md, раздел v5.")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")), must_exist=False)
    warn_no_stems(data)
    ckdir = ask_folder("Папка для чекпойнтов", str(last.get("ckdir_v5", "checkpoints/v5")), must_exist=False)
    last.update(data=data, ckdir_v5=ckdir)
    tail = ["--data", data, "--checkpoint_dir", ckdir, "--batch_size", "auto"]
    if resume:
        e = ask_int("Сколько эпох", int(last.get("v5_more", 5)))
        last["v5_more"] = e
        tail += ["--resume", "--epochs", str(e)]
        if ask_yes("Новое короткое расписание с малой скоростью обучения (дообучение, а не продолжение)", True):
            tail += ["--finetune"]
    else:
        k = ask_choice("Размер сети", [(n, h) for n, h, _ in V5_PRESETS], int(last.get("v5_preset", 2)))
        last["v5_preset"] = k
        e = ask_int("Эпох (остановится раньше, если 6 эпох нет улучшения)", int(last.get("v5_epochs", 40)))
        last["v5_epochs"] = e
        tail += ["--preset", V5_PRESETS[k - 1][2], "--epochs", str(e)]
    workers = ask_int("Процессов загрузки данных", int(last.get("workers", min(4, max(0, (os.cpu_count() or 2) - 1)))), 0, 32)
    last["workers"] = workers
    return launcher(last, "train_v5.py") + tail + ["--workers", str(workers)]


def step_critic(last: Dict[str, object]) -> Optional[List[str]]:
    title("Критик v5")
    print("Сеть-оценщик «похоже на работу человека?» для поиска при генерации (generate_v5.py --search 4).")
    print("Учится отличать карты людей от «пресных» (всё на 1/2, рис срезан), испорченных и карт самой сети.")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")), must_exist=False)
    ckdir = ask_folder("Папка чекпойнтов v5", str(last.get("ckdir_v5", "checkpoints/v5")), must_exist=False)
    last.update(data=data, ckdir_v5=ckdir)
    argv = [sys.executable, "train_critic.py", "--data", data, "--checkpoint_dir", ckdir]
    best = Path(ckdir) / "v5_best.pt"
    if best.exists() and ask_yes("Добавить карты самой сети v5 как отрицательные примеры (дольше, но критик точнее)", True):
        argv += ["--v5", str(best)]
    return argv


def sampling_argv(last: Dict[str, object]) -> List[str]:
    """Real maps only + how much more often the hard ones are drawn."""
    argv = ["--real_only"]
    print(color("  Только настоящие карты датасета: облегчённые и сгенерированные сетью карты (если они есть в датасете)"
                " и DPO не используются.", "2"))
    k = ask_choice("Как часто брать сложные карты (в датасете обычно перевес лёгких)",
                   [(n, h) for n, h, _, _ in SAMPLING], min(len(SAMPLING), int(last.get("sampling_i", 1))))
    last["sampling_i"] = k
    _, _, bal, tilt = SAMPLING[k - 1]
    return argv + ["--sr_balance", f"{bal:g}", "--hard_tilt", f"{tilt:g}"]


def step_train(last: Dict[str, object], resume: bool = False) -> Optional[List[str]]:
    title("Продолжить обучение" if resume else "Обучение сети")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")), must_exist=False)
    warn_no_stems(data)
    ckdir = ask_folder("Папка для чекпойнтов", str(last.get("ckdir", "checkpoints/hybrid")), must_exist=False)
    last.update(data=data, ckdir=ckdir)
    found = gpus()
    if found:
        print(color("Видеокарты: " + ", ".join(f"{i}: {n} ({m:.0f} ГБ)" for i, (n, m) in enumerate(found)), "32"))
    else:
        print(color("Видеокарта не найдена (или PyTorch без CUDA): обучение пойдёт на процессоре — очень медленно.", "33"))
    n_gpu = ask_int("Сколько видеокарт использовать", int(last.get("gpus", max(1, len(found)))), 1, max(1, len(found))) if len(found) > 1 else 1
    last["gpus"] = n_gpu
    argv_tail: List[str] = ["--data", data, "--checkpoint_dir", ckdir]
    if resume:
        argv_tail += ["--resume"]
        e = ask_int("Сколько эпох ещё", int(last.get("more_epochs", 10)))
        last["more_epochs"] = e
        argv_tail += ["--epochs", str(e)]
    else:
        k = ask_choice("Размер сети", [(n, h) for n, h, _, _ in PRESETS], int(last.get("preset_i", 2)))
        last["preset_i"] = k
        _, _, preset, batch = PRESETS[k - 1]
        mem = min((m for _, m in found[:n_gpu]), default=8.0)
        batch = max(2, int(batch * min(2.0, mem / 8.0)) // 2 * 2)
        L = ask_choice("Длительность", [(n, h) for n, h, _, _ in LENGTHS], int(last.get("length_i", 1)))
        last["length_i"] = L
        _, _, epochs, steps = LENGTHS[L - 1]
        if epochs == 0:
            epochs = ask_int("Эпох", int(last.get("epochs", 40)))
            steps = ask_int("Шагов в эпохе", int(last.get("steps", 1000)))
            last.update(epochs=epochs, steps=steps)
        argv_tail += ["--preset", preset, "--epochs", str(epochs), "--steps_per_epoch", str(steps)]
        if found and ask_choice("Размер батча", [("Подобрать автоматически", "рекомендуется: пробные шаги перед обучением, "
                                                   "занимает ~90 % памяти каждой видеокарты"),
                                                  ("Задать самому…", f"для {mem:.0f} ГБ примерно {batch}")],
                                1 if last.get("batch_auto", True) else 2) == 1:
            last["batch_auto"] = True
            frac = ask_float("Какую долю памяти занять (0.5-0.95)", float(last.get("mem_fraction", 0.9)))
            last["mem_fraction"] = frac
            argv_tail += ["--batch_size", "auto", "--mem_fraction", f"{frac:g}"]
        else:
            last["batch_auto"] = False
            b = ask_int(f"Размер батча на видеокарту (для {mem:.0f} ГБ подойдёт {batch}; чётный)", int(last.get("batch", batch)), 2, 512)
            b += b % 2
            last["batch"] = b
            argv_tail += ["--batch_size", str(b)]
    pat = ask_int("Остановить обучение, если столько эпох подряд нет улучшения на валидации (0 = все эпохи)",
                  int(last.get("patience", 5)), 0, 1000)
    last["patience"] = pat
    if pat:
        argv_tail += ["--patience", str(pat)]
    print(color(f"  Остановить вручную после текущей эпохи и перейти к следующему шагу: создать файл {Path(ckdir) / 'STOP'}", "2"))
    argv_tail += sampling_argv(last)
    workers = ask_int("Процессов загрузки данных", int(last.get("workers", min(4, max(0, (os.cpu_count() or 2) - 1)))), 0, 32)
    last["workers"] = workers
    argv_tail += ["--workers", str(workers)]
    if ask_choice("Техники обучения", [("Всё включено", "рекомендуется: все улучшения из HYBRID_GUIDE.md 4.9"),
                                        ("Настроить по отдельности…", "")], 1 if last.get("all_tech", True) else 2) == 1:
        last["all_tech"] = True
    else:
        last["all_tech"] = False
        off = dict(last.get("tech_off", {}))
        for key, q, hint, flags in TECHNIQUES:
            on = ask_yes(f"  {q}" + (color(f" ({hint})", "2") if hint else ""), not off.get(key, False))
            off[key] = not on
            if not on:
                argv_tail += flags
        last["tech_off"] = off
    if n_gpu > 1:
        argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}", "train_hybrid.py"] + argv_tail
    else:
        argv = [sys.executable, "train_hybrid.py"] + argv_tail
    return argv


def step_diagnose(last: Dict[str, object]) -> Optional[List[str]]:
    title("Диагностика звёзд")
    print("Сколько звёзд даёт сеть на запросы 2–8★: сырая карта, после подгонки и потолок. Пишет калибровку для генерации.")
    ck = ask_folder("Чекпойнт", str(last.get("checkpoint", "checkpoints/hybrid/hybrid_best.pt")), must_exist=False)
    songs = ask_folder("Папка с .osz (со звуком)", str(last.get("beatmaps", "sample beatmaps")))
    n = ask_int("Сколько песен проверить", int(last.get("diag_songs", 30)), 1)
    last.update(checkpoint=ck, beatmaps=songs, diag_songs=n)
    return [sys.executable, "sr_diagnose.py", "-c", ck, "--songs", songs, "--max_songs", str(n), "--write_calibration", "sr_cal.json"]


def step_finetune(last: Dict[str, object]) -> Optional[List[str]]:
    title("Дообучение на сложных")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")))
    ckdir = ask_folder("Папка чекпойнтов", str(last.get("ckdir", "checkpoints/hybrid")))
    sr = ask_float("Только карты от скольких звёзд", float(last.get("ft_sr", 5.0)))
    e = ask_int("Эпох", int(last.get("ft_epochs", 3)))
    last.update(data=data, ckdir=ckdir, ft_sr=sr, ft_epochs=e)
    print(color("  Только настоящие сложные карты датасета: сеть сравнивает себя с самими сложными картами, не с синтетикой.", "2"))
    return [sys.executable, "train_hybrid.py", "--data", data, "--checkpoint_dir", ckdir, "--resume", "--epochs", str(e),
            "--lr", "5e-5", "--finetune_min_sr", f"{sr:g}", "--real_only"]


def step_models(last: Dict[str, object]) -> Optional[List[str]]:
    title("Модель паттернов и библиотека мотивов")
    k = ask_choice("Что пересобрать", [("Модель паттернов", "learn_patterns.py: из подготовленного датасета"),
                                        ("Библиотеку мотивов", "mine_patterns.py: все повторяющиеся рисунки карт"),
                                        ("Обе", "")], int(last.get("models_i", 3)))
    last["models_i"] = k
    data = str(last.get("data", "dataset_hybrid"))
    beatmaps = str(last.get("beatmaps", "sample beatmaps"))
    a = [sys.executable, "learn_patterns.py", "--data", data]
    b = [sys.executable, "mine_patterns.py", "--input_dir", beatmaps]
    return a if k == 1 else (b if k == 2 else a + ["&&"] + b)


STEPS = [
    ("Полный цикл", "чистка → подготовка → обучение → диагностика"),
    ("Чистка архивов", "только нужное для обучения"),
    ("Подготовка датасета", "признаки аудио и параметры карт"),
    ("Обучение сети с нуля", ""),
    ("Продолжить обучение", "с последнего чекпойнта"),
    ("Диагностика звёзд", "где теряются звёзды"),
    ("Дообучение на сложных", "короткий второй этап только на сложных картах"),
    ("Модель паттернов / библиотека мотивов", ""),
    ("Рецепт v2 (контроль)", "то же обучение, что у v2: все добавки v3/v4 выключены"),
    ("Обучение v5", "новая сеть: ноты по очереди, слышит вокал/барабаны отдельно"),
    ("Дообучение v5", "продолжить или дообучить с малой скоростью"),
    ("Критик v5", "оценщик для поиска лучшего варианта фразы при генерации"),
]


def execute(argv: List[str]) -> bool:
    """Runs argv, which may chain several commands with '&&'."""
    cur: List[str] = []
    for a in argv + ["&&"]:
        if a == "&&":
            code = run(cur) if cur else 0
            if code == INTERRUPTED:
                print(color("\nШаг прерван (Ctrl+C).", "33"))
                if "train_hybrid.py" in cur or "train_v5.py" in cur:
                    print("Лучшая эпоха уже сохранена (hybrid_best.pt / v5_best.pt, по валидации), следующие шаги берут её.")
                if not ask_yes("Перейти к следующим шагам?", "train_hybrid.py" in cur or "train_v5.py" in cur):
                    return False
            elif code != 0:
                print(color("Шаг завершился с ошибкой: дальше не иду.", "31"))
                return False
            cur = []
        else:
            cur.append(a)
    return True


def main() -> None:
    _setup_console()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    print(color("AIManiaMapper — обучение нейросети", "1;35"))
    print(color("Enter = вариант в [скобках]. Ctrl+C — выход (во время шага — прервать шаг). Подробности: HYBRID_GUIDE.md, раздел 4.9.", "2"))
    while True:
        last = load_settings()
        try:
            k = ask_choice("\nЧто сделать?", STEPS, min(len(STEPS), int(last.get("step", 4))))
            last["step"] = k
            plan: List[List[str]] = []
            if k == 1:
                for fn in (step_clean, step_prepare, step_train, step_diagnose):
                    a = fn(last)
                    if a:
                        plan.append(a)
            else:
                fn = {2: step_clean, 3: step_prepare, 4: step_train, 5: lambda l: step_train(l, resume=True),
                      6: step_diagnose, 7: step_finetune, 8: step_models, 9: step_v2_recipe, 10: step_v5,
                      11: lambda l: step_v5(l, resume=True), 12: step_critic}[k]
                a = fn(last)
                if a:
                    plan.append(a)
            save_settings(last)
            if not plan:
                continue
            title("Итог")
            for a in plan:
                print("  " + command_text(a))
            if not ask_yes("Запустить?", True):
                continue
            ok = all(execute(a) for a in plan)
            print()
            print(color("Готово!" if ok else "Остановлено.", "1;32" if ok else "1;31") + " Те же шаги без вопросов:")
            for a in plan:
                print("  " + command_text(a))
        except KeyboardInterrupt:
            print("\nВыход.")
            return
        except Exception:
            print(color("Что-то пошло не так:", "31"))
            traceback.print_exc()
        if not ask_yes("\nСделать что-то ещё?", True):
            return


if __name__ == "__main__":
    main()
