#!/usr/bin/env python3
"""
Скрипт для создания и отправки Task в ClearML
"""
import json
import argparse
import sys
from pathlib import Path
from clearml import Task

def create_and_enqueue_task(config_path: str, queue_name: str = "default"):
    """
    Создает Task и отправляет его в очередь для выполнения агентом
    """
    # Загружаем конфигурацию
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Создаем Task
    task = Task.init(
        project_name='Video Processing',
        task_name=f"{config['corruption_type']}_s{config['severity']}",
        task_type=Task.TaskTypes.inference,
        auto_connect_frameworks={'matplotlib': False, 'tensorflow': False, 'pytorch': False}
    )
    
    # Сохраняем параметры в Task
    task.set_parameter('input_path', config['input_path'])
    task.set_parameter('output_path', config['output_path'])
    task.set_parameter('input_names', json.dumps(config['input_names']))
    task.set_parameter('corruption_type', config['corruption_type'])
    task.set_parameter('severity', str(config['severity']))
    
    # Указываем скрипт для выполнения
    # В простейшем случае - просто вызываем set_script() без аргументов
    # Он автоматически возьмет текущий запущенный скрипт
    task.set_script()
    
    # Отправляем в очередь
    print(f"Отправка задачи в очередь '{queue_name}'...")
    task.execute_remotely(queue_name=queue_name, exit_process=True)
    
    print(f"Task создан и отправлен в очередь '{queue_name}'")
    print(f"ID Task: {task.id}")
    print(f"Отслеживать выполнение можно по ссылке: {task.get_output_log_webpage()}")
    
    return task

def parse_args():
    parser = argparse.ArgumentParser(description="Отправка задачи в ClearML")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task_config.json",
        help="Путь к конфигурационному файлу"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="default",
        help="Имя очереди ClearML"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Проверяем существование конфига
    if not Path(args.config).exists():
        print(f"Ошибка: Конфигурационный файл '{args.config}' не найден!")
        sys.exit(1)
    
    task = create_and_enqueue_task(args.config, args.queue)