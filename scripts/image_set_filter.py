#!/usr/bin/env python3
"""
Упрощенный скрипт для тестирования ClearML агента
Добавляет шум к тестовому видео и сохраняет результат
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from imagecorruptions import corrupt
from clearml import Task

def main():
    print("🚀 Запуск упрощенного тестового скрипта")
    
    # СОЗДАЕМ Task для удаленного выполнения
    task = Task.init(
        project_name='Video Processing Test',
        task_name='Simple Test 100',
        task_type=Task.TaskTypes.inference,
        auto_connect_frameworks={'matplotlib': False, 'tensorflow': False, 'pytorch': False}
    )
    
    # ОТПРАВЛЯЕМ задачу в очередь для удаленного выполнения
    print("📤 Отправка задачи в очередь 'default'...")
    task.execute_remotely(queue_name='default', exit_process=True)
    
    # ВАЖНО: Код после execute_remotely() выполнится ТОЛЬКО на агенте!
    # Весь код ниже будет выполняться только когда агент подхватит задачу
    
    print("✅ Задача выполняется на агенте!")
    
    # Определяем пути (используем абсолютные пути от текущей директории)
    current_dir = Path("/home/sadmin/Work/projects/Mirror/video_filter_project/")  # поднимаемся на уровень выше scripts/
    input_path = current_dir / "data" / "input" / "example.mp4"
    output_path = current_dir / "data" / "output" / "example-result.mp4"
    
    print(f"📂 Текущая директория: {current_dir}")
    print(f"📂 Путь к входному файлу: {input_path}")
    print(f"📂 Путь к выходному файлу: {output_path}")
    
    # Проверяем существование входного файла
    if not input_path.exists():
        print(f"❌ Ошибка: Входной файл {input_path} не найден!")
        print("Содержимое директории data/input/:")
        input_dir = current_dir / "data" / "input"
        if input_dir.exists():
            for f in input_dir.glob("*"):
                print(f"  - {f.name}")
        else:
            print(f"  Директория {input_dir} не существует")
        sys.exit(1)
    
    # Создаем выходную директорию, если её нет
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📹 Начинаем обработку видео: {input_path.name}")
    print("⚙️ Параметры:")
    print("  - Тип шума: gaussian_noise")
    print("  - Уровень: 3")
    
    # Открываем видео
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Ошибка: Не удалось открыть видео {input_path}")
        sys.exit(1)
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Параметры видео:")
    print(f"  - Кадров: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Размер: {width}x{height}")
    
    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Ошибка: Не удалось создать выходное видео {output_path}")
        cap.release()
        sys.exit(1)
    
    # Обрабатываем кадры
    frame_count = 0
    print("🔄 Начинаем обработку кадров...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Добавляем шум
            corrupted = corrupt(frame, corruption_name='gaussian_noise', severity=3)
            
            # Записываем результат
            out.write(corrupted)
            
            frame_count += 1
            
            # Показываем прогресс каждый 10-й кадр
            if frame_count % 10 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"  Прогресс: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                # Логируем прогресс в ClearML
                task.get_logger().report_scalar(
                    title="Progress",
                    series="frames",
                    value=frame_count,
                    iteration=frame_count
                )
    
    except Exception as e:
        print(f"❌ Ошибка при обработке: {e}")
        raise
    
    finally:
        # Освобождаем ресурсы
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"✅ Обработка завершена!")
    print(f"📁 Результат сохранен: {output_path}")
    
    # Проверяем размер выходного файла
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"📊 Размер файла: {size_mb:.2f} MB")
    
    # Загружаем результат как артефакт в ClearML
    print("📤 Загружаем результат как артефакт в ClearML...")
    task.upload_artifact(
        name='processed_video',
        artifact_object=str(output_path)
    )
    print("✅ Артефакт загружен")
    
    # Закрываем Task
    task.close()
    print("🏁 Задача завершена")

if __name__ == "__main__":
    main()