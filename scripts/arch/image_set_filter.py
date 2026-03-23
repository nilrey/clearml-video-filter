#!/usr/bin/env python3
from pathlib import Path
import json
from typing import Any
import argparse
import os

import cv2
from clearml import Task
from imagecorruptions import corrupt


ALLOWED_CORRUPTION_NAMES = [
    "gaussian_noise",
    "zoom_blur",
    "jpeg_compression",
    "brightness",
    "saturate",
    "spatter",
    "speckle_noise",
    "impulse_noise",
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "contrast",
]


def add_filter(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    corruption_name: str = "gaussian_noise",
    severity: int = 1,
) -> None:

    task = Task.current_task()
    logger = task.get_logger() if task is not None else None

    if input_path is None or output_path is None:
        raise ValueError(
            "input_path и output_path должны быть заданы"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)

    if corruption_name not in ALLOWED_CORRUPTION_NAMES:
        raise ValueError(
            f"Некорректное значение corruption_name='{corruption_name}'. "
            f"Допустимые значения: {', '.join(ALLOWED_CORRUPTION_NAMES)}"
        )

    try:
        severity_int = int(severity)
    except (TypeError, ValueError):
        raise ValueError("Не возможно преобразовать в число. severity должно быть числом от 1 до 5")

    if not 1 <= severity_int <= 5:
        raise ValueError("severity должно быть числом от 1 до 5")

    if not input_path.exists():
        raise FileNotFoundError(f"input_path не существует: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть входной файл: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать выходной файл: {output_path}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            corrupted = corrupt(
                frame,
                corruption_name=corruption_name,
                severity=severity_int,
            )
            out.write(corrupted)
            frame_count += 1
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Добавим метрику количества кадров
    if logger is not None:
        logger.report_single_value("frame_count", frame_count)

    print("Скрипт успешно завершен")


def _init_or_get_task(
    project_name: str,
    task_name: str,
) -> Task:
    existing = Task.current_task()
    task = existing
    if task is None:
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=Task.TaskTypes.inference,
            reuse_last_task_id=True,
            # Важно: не даём ClearML auto-connect к argparse затирать hyperparameters
            auto_connect_arg_parser=False,
        )
    return task


def _get_task_parameters_flat(task: Task | None) -> dict[str, Any]:
    # hack т.к. параметры в Task имеют вид 'Section/name' переделаем в нормальный вид - 'name'.
    if task is None:
        return {}
    try:
        raw = task.get_parameters() or {}
    except Exception:
        return {}

    flat: dict[str, Any] = {}
    for key, value in raw.items(): 
        name = key.split("/", 1)[-1] 
        if name == "input_names" and isinstance(value, str): 
            try:
                parsed = json.loads(value)
                value = parsed
            except Exception:
                pass
        flat[name] = value
    return flat


def _parse_input_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]
    return [str(value)]


def _set_general_params(task: Task, params: dict[str, Any]) -> None:
    """
    Обновляет параметры в секции General точечно (не перезаписывая всю секцию),
    чтобы не "очищать" значения при частичных апдейтах.
    """
    for k, v in params.items():
        if v is None:
            continue
        name = "General/" + str(k)
        if k == "input_names":
            if isinstance(v, list):
                value_str = json.dumps(v, ensure_ascii=False)
            else:
                value_str = str(v)
        else:
            value_str = str(v)
        task.set_parameter(name=name, value=value_str)


def run_from_task_params(task: Task, cli_overrides: dict[str, Any]) -> None:
    task_params = _get_task_parameters_flat(task)

    # Берём сохранённые hyperparameters, а CLI перекрывает только переданные значения
    merged: dict[str, Any] = dict(task_params)
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v

    # JSON в list[str]
    merged["input_names"] = _parse_input_names(merged.get("input_names"))

    print("Получены параметры:")
    for k in sorted(merged.keys()):
        print(f"{k}: {merged[k]}")

    input_root = merged.get("input_path")
    output_root = merged.get("output_path")
    input_names = merged.get("input_names")
    corruption_name = merged.get("corruption_name", "gaussian_noise")
    severity = merged.get("severity", 1)

    if input_root is None :
        raise ValueError("Параметры Task должны содержать input_path")
    if output_root is None:
        raise ValueError("Параметры Task должны содержать output_path")
    if input_names is None:
        raise ValueError("Параметры Task должны содержать input_names (список файлов)")

    # Сохраняем в Task только то, что явно передали в CLI (override),
    # иначе рискуем перезаписать/очистить параметры при запуске на агенте.
    to_persist: dict[str, Any] = {}
    for k, v in cli_overrides.items():
        if v is None:
            continue
        if k == "input_names":
            to_persist[k] = _parse_input_names(v)
        else:
            to_persist[k] = v
    if to_persist:
        _set_general_params(task, to_persist)

    input_root_p = Path(input_root)
    output_root_p = Path(output_root)

    for name in input_names:
        in_file = input_root_p / str(name)
        out_file = output_root_p / str(name)
        add_filter(
            input_path=in_file,
            output_path=out_file,
            corruption_name=str(corruption_name),
            severity=int(severity),
        )


if __name__ == "__main__":
    print("Начало работы скрипта")
    p = argparse.ArgumentParser()
    p.add_argument("--project-name")
    p.add_argument("--task-name")

    p.add_argument("--input-path")
    p.add_argument("--output-path")
    p.add_argument("--input-names")
    p.add_argument("--corruption-name")
    p.add_argument("--severity")
    args = p.parse_args()

    current = Task.current_task()
    if current is not None:
        task = current
    else:
        try:
            if args.project_name and args.task_name:
                task = _init_or_get_task(project_name=args.project_name, task_name=args.task_name)
            else:
                # Запуск из ClearML Agent: привязываемся к Task по env CLEARML_TASK_ID
                task_id = os.environ.get("CLEARML_TASK_ID")
                if not task_id:
                    p.error("the following arguments are required for local run: --project-name, --task-name")
                task = Task.init(
                    continue_last_task=task_id,
                    auto_connect_arg_parser=False,
                )
        except Exception:
            p.error("the following arguments are required for local run: --project-name, --task-name")

    cli_overrides: dict[str, Any] = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "input_names": args.input_names,
        "corruption_name": args.corruption_name,
        "severity": args.severity,
    }

    run_from_task_params(task, cli_overrides=cli_overrides)

