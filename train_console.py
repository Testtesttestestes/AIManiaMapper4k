#!/usr/bin/env python3
"""
Консольная программа для обучения: чистка датасета, подготовка, обучение сети, диагностика, самообучение - по шагам.

    python train_console.py            (на Windows: двойной клик по train_console.bat)

Как mapper_console.py: простые вопросы с вариантами по умолчанию в [скобках], итог перед запуском, та же команда в
конце (её можно сохранить и запускать без вопросов), выбор запоминается до следующего раза. Несколько видеокарт
находятся сами, обучение на них запускается через torchrun.
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
    ("rare", "Больший вес редких сложностей", "сложные карты весят больше в потерях", ["--sr_rare", "0", "--sr_balance", "0"]),
    ("patterns", "Знания паттернов и библиотеки мотивов", "структура карты и как реальные карты продолжают аккорды",
     ["--pattern_weight", "0", "--prior_weight", "0", "--plan_prob", "0"]),
    ("plan", "План по тактам («сначала подумать»)", "техника каждого такта как цель и как вход", ["--plan_prob", "0"]),
    ("fix", "«Сломай и почини» (самокоррекция)", "сеть учится находить и исправлять ошибки", ["--corrupt_prob", "0"]),
    ("canvas", "Холст карты (дорисовка с оглядкой на уже поставленное)", "", ["--canvas_prob", "0", "--corrupt_prob", "0"]),
    ("ladder", "Лестница сложностей", "какие удары остаются на всех сложностях, какие только на сложных", ["--ladder_weight", "0"]),
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
    argv = [sys.executable, "prepare_hybrid_dataset.py", "--input_dir", folder, "--output_dir", out]
    if ask_yes("Добавить облегчённые версии сложных карт (прореживание)?", bool(last.get("derived", True))):
        last["derived"] = True
        return argv + ["&&", sys.executable, "make_derived_maps.py", "--data", out]
    last["derived"] = False
    return argv


def step_train(last: Dict[str, object], resume: bool = False) -> Optional[List[str]]:
    title("Продолжить обучение" if resume else "Обучение сети")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")), must_exist=False)
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


def step_self(last: Dict[str, object]) -> Optional[List[str]]:
    title("Самообучение по награде + DPO")
    print("Сеть делает несколько вариантов для звёзд, где карт мало; лучшие по награде попадают в датасет с их настоящими")
    print("звёздами, плюс пары «человек лучше сети» для DPO. Потом второй раунд обучения на одной видеокарте.")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")), must_exist=False)
    ck = ask_folder("Чекпойнт", str(last.get("checkpoint", "checkpoints/hybrid/hybrid_best.pt")), must_exist=False)
    targets = ask("Звёзды (через запятую)", str(last.get("self_targets", "5.5,6.5,7.5,8.5")))
    cands = ask_int("Вариантов на песню и звёзды", int(last.get("self_cands", 4)), 1, 8)
    epochs = ask_int("Эпох второго раунда (0 = только сгенерировать)", int(last.get("self_epochs", 5)), 0)
    last.update(data=data, checkpoint=ck, self_targets=targets, self_cands=cands, self_epochs=epochs)
    argv = [sys.executable, "make_self_maps.py", "--data", data, "--checkpoint", ck, "--targets", targets,
            "--candidates", str(cands)]
    if epochs > 0:
        argv += ["&&", sys.executable, "train_hybrid.py", "--data", data, "--checkpoint_dir", str(Path(ck).parent), "--resume",
                 "--epochs", str(epochs), "--dpo_weight", "0.1"]
    return argv


def step_finetune(last: Dict[str, object]) -> Optional[List[str]]:
    title("Дообучение на сложных")
    data = ask_folder("Папка датасета", str(last.get("data", "dataset_hybrid")))
    ckdir = ask_folder("Папка чекпойнтов", str(last.get("ckdir", "checkpoints/hybrid")))
    sr = ask_float("Только карты от скольких звёзд", float(last.get("ft_sr", 5.0)))
    e = ask_int("Эпох", int(last.get("ft_epochs", 3)))
    last.update(data=data, ckdir=ckdir, ft_sr=sr, ft_epochs=e)
    return [sys.executable, "train_hybrid.py", "--data", data, "--checkpoint_dir", ckdir, "--resume", "--epochs", str(e),
            "--lr", "5e-5", "--finetune_min_sr", f"{sr:g}"]


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
    ("Полный цикл", "чистка → подготовка → обучение → диагностика → самообучение"),
    ("Чистка архивов", "только нужное для обучения"),
    ("Подготовка датасета", "признаки аудио и параметры карт"),
    ("Обучение сети с нуля", ""),
    ("Продолжить обучение", "с последнего чекпойнта"),
    ("Диагностика звёзд", "где теряются звёзды"),
    ("Самообучение по награде + DPO", "сложные карты, которых не хватает"),
    ("Дообучение на сложных", "короткий второй этап только на сложных картах"),
    ("Модель паттернов / библиотека мотивов", ""),
]


def execute(argv: List[str]) -> bool:
    """Runs argv, which may chain several commands with '&&'."""
    cur: List[str] = []
    for a in argv + ["&&"]:
        if a == "&&":
            code = run(cur) if cur else 0
            if code == INTERRUPTED:
                print(color("\nШаг прерван (Ctrl+C).", "33"))
                if "train_hybrid.py" in cur:
                    print("Лучшая эпоха уже сохранена в hybrid_best.pt (по валидации), следующие шаги берут её.")
                if not ask_yes("Перейти к следующим шагам?", "train_hybrid.py" in cur):
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
            k = ask_choice("\nЧто сделать?", STEPS, int(last.get("step", 4)))
            last["step"] = k
            plan: List[List[str]] = []
            if k == 1:
                for fn in (step_clean, step_prepare, step_train, step_diagnose, step_self):
                    a = fn(last)
                    if a:
                        plan.append(a)
            else:
                fn = {2: step_clean, 3: step_prepare, 4: step_train, 5: lambda l: step_train(l, resume=True),
                      6: step_diagnose, 7: step_self, 8: step_finetune, 9: step_models}[k]
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
