#!/usr/bin/env python3
from pathlib import Path
import subprocess
import argparse
import shutil
import cv2
from clearml import Task, Dataset
from imagecorruptions import corrupt
from datetime import datetime
import json

# Параметры ClearML задачи 
TASK_NAME = "Dataset Filter"
TASK_PROJECT = "Initial Scripts"
OUT_DATASET_PROJECT_NAME = "MIRROR"
REQUIREMENTS_FILE = "requirements.txt"

# Наложение фильтров
# список допустимых искажений
ALLOWED_CORRUPTION_NAMES = [
    "gaussian_noise", "zoom_blur", "jpeg_compression", "brightness", "saturate",
    "spatter", "speckle_noise", "impulse_noise", "shot_noise", "defocus_blur",
    "motion_blur", "contrast"
]
# default values
DEFAULT_SEVERITY = 5
DEFAULT_CORRUPTION_NAME = "gaussian_noise"

# Paths
BUFFER_DIR = '/home/sadmin/Documents/Mirror/data/output'
# BUFFER_DIR = '/home/ubuntu/mirror_storage/storage/data/output'
# BUFFER_DIR = 'D:/Tmp/Mirror/data/output'



def add_filter(input_path=None, output_path=None, corruption_name="gaussian_noise", severity=5):
    if input_path is None or output_path is None:
        raise ValueError("input_path и output_path должны быть заданы")
    
    input_path = Path(input_path)
    output_path = Path(output_path)

    temp_output = output_path.with_suffix(".tmp.mp4")
    
    if corruption_name not in ALLOWED_CORRUPTION_NAMES:
        raise ValueError(f"Некорректное имя искажения: {corruption_name}")
    try:
        severity = int(severity)
    except Exception:
        raise ValueError("severity must be integer")
    if not 1 <= severity <= 5:
        raise ValueError("severity 1..5")

    #читаем файлы
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"НCannot open {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #создаем выходной файл во временной папке
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"НCannot create {output_path}")

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

    if frame_count == 0:
        raise RuntimeError(f"File not created or empty: {temp_output}")

    # перекодировка в H.264
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_output),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error occurred running ffmpeg: {e}")

    # удаляем временный файл
    temp_output.unlink(missing_ok=True)

    return frame_count


def get_parameter_from_task(task, param_name, default_value=None):
    # выбираем параметры из секции General
    if task is None:
        return default_value
    
    try:
        param_value = task.get_parameter(f"General/{param_name}")
        if param_value is not None:
            print(f"Param: {param_name} from Task: {param_value}")
            return param_value
    except Exception:
        pass
    
    return default_value


def save_task_parameters(task, hyperparams):
    if task is None:
        return
    
    # Сохраняем параметры в секцию General
    for param_name, param_value in hyperparams.items():
        if param_value is not None:
            task.set_parameter(f"General/{param_name}", str(param_value))

    """
    Отладочный блок. 
    Сделана имитация наличия конфигурационных объектов. При инициализации задачи, конфигурационных объектов нет. 
    Сохраняем input_dataset как строку в формате JSON
    """
    # input_dataset = json.dumps({"id": None, "name": None, "location": None}, ensure_ascii=False)
    # algorithm = json.dumps({ "name": None, "id": None }, ensure_ascii=False)
    # input_params = json.dumps({ "corruption_name": None, "severity": None }, ensure_ascii=False)
    # task.set_configuration_object("input_dataset", input_dataset)
    # task.set_configuration_object("algorithm", algorithm)
    # task.set_configuration_object("input_params", input_params)


def get_config_values(task, config_name):
    config_string = task.get_configuration_object(config_name)
    if config_string:
        try:
            config_params = json.loads(config_string)
            return config_params
        except json.JSONDecodeError:
            raise ValueError(f"json.loads вызвал ошибку конфигурационного объекта: {config_name}")
    return None


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
        print("Args received. Create new task")
        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name
        )
        # Указываем зависимости из requirements.txt
        # task.add_requirements(REQUIREMENTS_FILE)
    else:
        print("No args, use existed task")
        # Если нет аргументов, берем существующую задачу или создаем с временными именами
        task = Task.current_task()
        if task is None:
            print("Noexisted task, create new")
            # Создаем задачу с временными именами, которые потом будут переопределены параметрами
            task = Task.init(
                project_name="Temporary Project",
                task_name="Temporary Task"
            )
            # Указываем зависимости из requirements.txt
            # task.add_requirements(REQUIREMENTS_FILE)
    
    # Извлекаем и выводим значения конфигурации
    input_dataset_cfg = get_config_values(task, "input_dataset")
    # algorithm_cfg = get_config_values(task, "algorithm")
    input_params_cfg = get_config_values(task, "input_params")

    if not input_dataset_cfg:
        input_dataset_cfg = {"id": None, "name": None}
    if not input_params_cfg:
        input_params_cfg = {"severity": None, "corruption_name": None}

    # Получаем параметры с приоритетом: Task config параметры > Task General параметры > аргументы командной строки > значения по умолчанию
    project_name = get_parameter_from_task(task, "project_name") or args.project_name or TASK_PROJECT
    task_name = task.name #get_parameter_from_task(task, "task_name") or args.task_name or TASK_NAME
    input_dataset_id = input_dataset_cfg.get("id") or get_parameter_from_task(task, "input_dataset_id") or args.input_dataset_id
    corruption_name = input_params_cfg.get("corruption_name") or get_parameter_from_task(task, "corruption_name") or args.corruption_name or DEFAULT_CORRUPTION_NAME
    severity = input_params_cfg.get("severity") or get_parameter_from_task(task, "severity") or args.severity or DEFAULT_SEVERITY
    # out_dataset_name = input_dataset_cfg.get("name") or f"Out_Dataset_" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Получаем имя входного датасета для формирования имени выходного датасета
    dataset = Dataset.get(dataset_id=input_dataset_id)
    out_dataset_name = f"{task_name} @ {dataset.name}"

    # Проверяем только input_dataset_id, так как он критичен
    if not input_dataset_id:
        raise ValueError("input_dataset_id must be provided either as Task parameter or command line argument")
    
    # Преобразуем severity в int
    try:
        severity = int(severity)
    except (ValueError, TypeError):
        raise ValueError(f"severity must be a numeric: {severity}")
    
    # Сохраняем параметры в Task для будущих запусков
    hyperparams = {
        "project_name": project_name,
        "task_name": task_name,
        "input_dataset_id": input_dataset_id,
        "corruption_name": corruption_name,
        "severity": severity
    }
    save_task_parameters(task, hyperparams)
    
    print(f"Task params:")
    print(f"Project name: {project_name}")
    print(f"Task name: {task_name}")
    print(f"Task ID: {task.id}")
    print(f"Input dataset ID: {input_dataset_id}")
    print(f"Corruption name: {corruption_name}")
    print(f"Severity: {severity}")

    input_dataset = Dataset.get(dataset_id=input_dataset_id)
    input_root = Path(input_dataset.get_local_copy())

    input_files = [p for p in input_root.iterdir() if p.is_file() ]
    if not input_files:
        raise RuntimeError("No files in input dataset")

    # создаем buffer_root если он не существует
    buffer_root = Path(BUFFER_DIR)
    buffer_root.mkdir(parents=True, exist_ok=True)
            
    # сохраняем файлы в buffer_root
    for p in input_files:
        out_file = buffer_root / p.name
        frame_count = add_filter(p, out_file, corruption_name, severity)
        if frame_count == 0:
            raise RuntimeError(f"Файл не создан или пуст: {out_file}")
    
    dataset = Dataset.create(dataset_project=OUT_DATASET_PROJECT_NAME, dataset_name=out_dataset_name)
    for f in sorted(buffer_root.iterdir()):
        if f.is_file():
            dataset.add_files(path=str(f))
    dataset.upload()
    dataset.finalize()
    if dataset.id:
        print("Dataset created success", dataset.id)
        for f in buffer_root.iterdir():
            if f.is_file():
                f.unlink()
    else:
        print(f"Error on dataset creation, dataset id is empty: {dataset.id}")
        
    task.upload_artifact('output_dataset', 
                    {'id': dataset.id, 'name': dataset.name}, 
                    metadata={'id': dataset.id, 'name': dataset.name})

if __name__ == '__main__':
    main()