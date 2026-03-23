#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import cv2
from clearml import Task, Dataset
from imagecorruptions import corrupt

# список допустимых искажений, 
ALLOWED_CORRUPTION_NAMES = [
    "gaussian_noise", "zoom_blur", "jpeg_compression", "brightness", "saturate",
    "spatter", "speckle_noise", "impulse_noise", "shot_noise", "defocus_blur",
    "motion_blur", "contrast"
]

# директория для временных файлов
BUFFER_DIR = '/home/sadmin/Documents/Mirror/data/output'
OUTPUT_DATASETS_NAME = "Output Datasets"
# BUFFER_DIR = '/home/ubuntu/mirror_storage/storage/data/output'
# OUTPUT_DATASETS_NAME = "MIRROR"

def add_filter(input_path=None, output_path=None, corruption_name="gaussian_noise", severity=5):
    if input_path is None or output_path is None:
        raise ValueError("input_path и output_path должны быть заданы")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if corruption_name not in ALLOWED_CORRUPTION_NAMES:
        raise ValueError(f"Некорректное имя искажения: {corruption_name}")
    try:
        severity = int(severity)
    except Exception:
        raise ValueError("severity дожно быть числом")
    if not 1 <= severity <= 5:
        raise ValueError("severity 1..5")

    #читаем файлы
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #создаем выходной файл
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать {output_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = corrupt(frame, corruption_name=corruption_name, severity=severity)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return frame_count


def get_parameter_from_task(task, param_name, default_value=None):
    """Получает параметр из Task Hyperparameters -> General"""
    if task is None:
        return default_value
    
    try:
        # Пытаемся получить параметр из секции General
        param_value = task.get_parameter(f"General/{param_name}")
        if param_value is not None:
            print(f"Используем параметр {param_name} из Task: {param_value}")
            return param_value
    except Exception:
        pass
    
    return default_value


def save_task_parameters(task, hyperparams):
    """Сохраняет параметры в Task"""
    if task is None:
        return
    
    # Сохраняем параметры в секцию General
    for param_name, param_value in hyperparams.items():
        if param_value is not None:
            task.set_parameter(f"General/{param_name}", str(param_value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", required=False, help="Project name (optional if set in Task)")
    parser.add_argument("--task-name", required=False, help="Task name (optional if set in Task)")
    parser.add_argument("--input-dataset-id", required=False, help="Input dataset ID (optional if set in Task)")
    parser.add_argument("--corruption-name", default=None, help="Corruption name (optional if set in Task)")
    parser.add_argument("--severity", default=None, help="Severity (optional if set in Task)")
    args = parser.parse_args()
    
    # Проверяем, есть ли аргументы командной строки
    has_cli_args = any([args.project_name, args.task_name, args.input_dataset_id])
    
    # Инициализируем задачу ClearML
    if has_cli_args:
        # Если есть аргументы, используем их для создания задачи
        task = Task.init(
            project_name=args.project_name, 
            task_name=args.task_name, 
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False
        )
    else:
        # Если нет аргументов, пытаемся получить существующую задачу или создаем с временными именами
        task = Task.current_task()
        if task is None:
            # Создаем задачу с временными именами, которые потом будут переопределены параметрами
            task = Task.init(
                project_name="Temporary Project", 
                task_name="Temporary Task", 
                auto_connect_arg_parser=False,
                auto_connect_frameworks=False
            )
    
    # Получаем параметры с приоритетом: Task параметры > аргументы командной строки > значения по умолчанию
    project_name = get_parameter_from_task(task, "project_name") or args.project_name or "Initial Scripts"
    task_name = get_parameter_from_task(task, "task_name") or args.task_name or "Dataset Filter"
    input_dataset_id = get_parameter_from_task(task, "input_dataset_id") or args.input_dataset_id
    corruption_name = get_parameter_from_task(task, "corruption_name") or args.corruption_name or "gaussian_noise"
    severity = get_parameter_from_task(task, "severity") or args.severity or 2
    
    # Проверяем только input_dataset_id, так как он критичен
    if not input_dataset_id:
        raise ValueError("input_dataset_id must be provided either as Task parameter or command line argument")
    
    # Преобразуем severity в int
    try:
        severity = int(severity)
    except (ValueError, TypeError):
        raise ValueError(f"severity должно быть числом, получено: {severity}")
    
    # Сохраняем параметры в Task для будущих запусков
    hyperparams = {
        "project_name": project_name,
        "task_name": task_name,
        "input_dataset_id": input_dataset_id,
        "corruption_name": corruption_name,
        "severity": severity
    }
    save_task_parameters(task, hyperparams)
    
    # Обновляем имя задачи, если оно изменилось
    if task.name != task_name or task.project != project_name:
        task.set_name(task_name)
        # Примечание: project нельзя изменить после создания, но для будущих запусков сохраним параметр
    
    print(f"Параметры запуска:")
    print(f"  Project name: {project_name}")
    print(f"  Task name: {task_name}")
    print(f"  Task ID: {task.id}")
    print(f"  Input dataset ID: {input_dataset_id}")
    print(f"  Corruption name: {corruption_name}")
    print(f"  Severity: {severity}")

    input_dataset = Dataset.get(dataset_id=input_dataset_id)
    input_root = Path(input_dataset.get_local_copy())

    input_files = [p for p in input_root.iterdir() if p.is_file() ]
    if not input_files:
        raise RuntimeError("Нет видеофайлов в input dataset")

    # создаем buffer_root если он не существует
    buffer_root = Path(BUFFER_DIR)
    buffer_root.mkdir(parents=True, exist_ok=True)
    
    # сохраняем файлы в buffer_root
    for p in input_files:
        out_file = buffer_root / p.name
        frame_count = add_filter(p, out_file, corruption_name, severity)
        if frame_count == 0:
            raise RuntimeError(f"Файл не создан или пуст: {out_file}")

    dataset_name = f"output_{input_dataset.name}_{corruption_name}_{severity}"
    dataset = Dataset.create(dataset_project=OUTPUT_DATASETS_NAME, dataset_name=dataset_name)
    for f in sorted(buffer_root.iterdir()):
        if f.is_file():
            dataset.add_files(path=str(f))
    dataset.upload()
    dataset.finalize()
    print("Датасет успешно создан", dataset.id)


if __name__ == '__main__':
    main()