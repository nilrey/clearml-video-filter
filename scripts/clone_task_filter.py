#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from typing import Any

from clearml import Task


def _parse_input_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # set JSON to list string
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]
    return [str(value)]


def _get_parent_params_flat(parent: Task) -> dict[str, Any]:
    """
    ClearML stores parameters as 'Section/name' (e.g. 'General/input_path').
    We flatten them into plain 'name' keys.
    """
    raw = parent.get_parameters() or {}
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        name = key.split("/", 1)[-1]
        if name == "input_names":
            parsed = _parse_input_names(value)
            if parsed is not None:
                value = parsed
        flat[name] = value
    return flat


def _set_general_params(task: Task, params: dict[str, Any]) -> None:
    """
    Обновляем параметры в секции General точечно (не перезаписывая секцию целиком),
    чтобы не "очищать" значения при частичных апдейтах.
    """
    for k, v in params.items():
        if v is None:
            continue
        name = f"General/{k}"
        if k == "input_names":
            if isinstance(v, list):
                value_str = json.dumps(v, ensure_ascii=False)
            else:
                value_str = str(v)
        else:
            value_str = str(v)
        task.set_parameter(name=name, value=value_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--clone-task-name")
    p.add_argument("--project-name")

    p.add_argument("--input-path")
    p.add_argument("--output-path")
    p.add_argument(
        "--input-names",
        default=None,
        help='Override input_names (JSON list like ["a.mp4"] or single name)',
    )
    p.add_argument("--corruption-name")
    p.add_argument("--severity")

    p.add_argument("--queue-name")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    parent = Task.get_task(task_id=args.task_id)
    parent_params = _get_parent_params_flat(parent)

    # Resolve clone name
    if args.clone_task_name:
        clone_name = args.clone_task_name
    else:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clone_name = f"{parent.name} Clone {ts}"

    # Resolve parameters (override -> else parent)
    input_path = args.input_path if args.input_path is not None else parent_params.get("input_path")
    output_path = args.output_path if args.output_path is not None else parent_params.get("output_path")

    if args.input_names is not None:
        input_names = _parse_input_names(args.input_names)
    else:
        input_names = _parse_input_names(parent_params.get("input_names"))

    corruption_name = args.corruption_name if args.corruption_name is not None else parent_params.get("corruption_name")

    severity = args.severity if args.severity is not None else parent_params.get("severity")
    if severity is not None:
        try:
            severity = int(severity)
        except Exception:
            # Keep original value if cannot cast
            pass

    if args.project_name:
        # Task.clone ожидает project id, поэтому резолвим id по имени проекта
        project_id = Task.get_project_id(args.project_name)
    else:
        project_id = parent.project

    # Create clone (Draft)
    cloned = Task.clone(
        source_task=parent,
        name=clone_name,
        parent=parent.id,
        project=project_id,
    )

    to_set: dict[str, Any] = {}
    if input_path is not None:
        to_set["input_path"] = input_path
    if output_path is not None:
        to_set["output_path"] = output_path
    if input_names is not None:
        to_set["input_names"] = input_names
    if corruption_name is not None:
        to_set["corruption_name"] = str(corruption_name)
    if severity is not None:
        to_set["severity"] = severity

    _set_general_params(cloned, to_set)

    # Enqueue cloned task
    Task.enqueue(cloned, queue_name=args.queue_name)

    print(f"Parent Task id: {parent.id}")
    print(f"Cloned Task id: {cloned.id}")
    try:
        print(f"Cloned Task URL: {cloned.get_output_log_web_page()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

