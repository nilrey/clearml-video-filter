#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from typing import Any
from pathlib import Path

from clearml import Task

REQUIREMENTS_FILE = "requirements.txt"


def _get_parent_params_flat(parent: Task) -> dict[str, Any]:
    raw = parent.get_parameters() or {}
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        name = key.split("/", 1)[-1]
        flat[name] = value
    return flat


def _set_general_params(task: Task, params: dict[str, Any]) -> None:
    for k, v in params.items():
        if v is None:
            continue
        name = f"General/{k}"
        if isinstance(v, (list, dict)):
            value_str = json.dumps(v, ensure_ascii=False)
        else:
            value_str = str(v)
        task.set_parameter(name=name, value=value_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clone ClearML task with dataset filter parameters"
    )
    p.add_argument("--task-id")
    p.add_argument("--clone-task-name")
    p.add_argument("--project-name")
    p.add_argument("--input-dataset-id")
    p.add_argument("--corruption-name")
    p.add_argument("--severity", type=int)
    p.add_argument("--queue-name")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Получаем исходную задачу
    parent = Task.get_task(task_id=args.task_id)
    parent_params = _get_parent_params_flat(parent)

    # Определяем имя для клонированной задачи
    if args.clone_task_name:
        clone_name = args.clone_task_name
    else:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clone_name = f"{parent.name} Clone {ts}"

    # Определяем параметры (переопределенные -> значения из родительской задачи)
    input_dataset_id = args.input_dataset_id if args.input_dataset_id is not None else parent_params.get("input_dataset_id")
    corruption_name = args.corruption_name if args.corruption_name is not None else parent_params.get("corruption_name")
    severity = args.severity if args.severity is not None else parent_params.get("severity")

    # Преобразуем severity в int, если это возможно
    if severity is not None and not isinstance(severity, int):
        try:
            severity = int(severity)
        except (ValueError, TypeError):
            # Если не удалось преобразовать, оставляем как есть
            pass

    # Определяем проект для клонированной задачи
    if args.project_name:
        project_id = Task.get_project_id(args.project_name)
    else:
        project_id = parent.project

    # Создаем клон (в статусе Draft)
    cloned = Task.clone(
        source_task=parent,
        name=clone_name,
        parent=parent.id,
        project=project_id,
    )
    # Указываем зависимости из requirements.txt
    cloned.add_requirements(REQUIREMENTS_FILE)

    # Собираем параметры для переопределения
    to_set: dict[str, Any] = {}
    if input_dataset_id is not None:
        to_set["input_dataset_id"] = input_dataset_id
    if corruption_name is not None:
        to_set["corruption_name"] = corruption_name
    if severity is not None:
        to_set["severity"] = severity

    # Применяем переопределенные параметры
    if to_set:
        _set_general_params(cloned, to_set)
        print("Updated parameters:")
        for key, value in to_set.items():
            print(f"  {key}: {value}")

    # Ставим клонированную задачу в очередь
    if args.queue_name:
        Task.enqueue(cloned, queue_name=args.queue_name)
        print(f"Task enqueued to queue: {args.queue_name}")
    else:
        print("Task created but not enqueued (no queue specified)")

    # Выводим информацию о созданной задаче
    print(f"\nSource Task id: {parent.id}")
    print(f"Source Task name: {parent.name}")
    print(f"Cloned Task id: {cloned.id}")
    print(f"Cloned Task name: {cloned.name}")
    try:
        print(f"Cloned Task URL: {cloned.get_output_log_web_page()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()