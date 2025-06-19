import cv2
from ultralytics import YOLO


def process_video(video_path):
    # Загрузка моделей
    model_plate = YOLO('plate_detection.pt')  # Убедитесь, что пути к моделям верные
    model_symbols = YOLO('best.pt')

    # Инициализация видеопотока
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка загрузки видео")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Обработка каждого кадра
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция номерного знака
        plates = model_plate(frame)

        if len(plates[0].boxes) == 0:
            # Номер не найден - пропускаем кадр
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Обработка всех обнаруженных номеров
        for box in plates[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Рисуем рамку номера на кадре
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Вырезаем номер
            plate_roi = frame[y1:y2, x1:x2]

            # Распознавание символов
            characters = model_symbols(plate_roi)
            detected_chars = []

            for char_box in characters[0].boxes:
                x1c, y1c, x2c, y2c = map(int, char_box.xyxy[0])
                char = model_symbols.names[int(char_box.cls)]
                confidence = char_box.conf.item()

                # Рисуем символы на вырезанном номере (для демонстрации)
                cv2.rectangle(plate_roi, (x1c, y1c), (x2c, y2c), (0, 0, 255), 1)
                cv2.putText(plate_roi, f"{char}", (x1c, y1c - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                detected_chars.append((x1c, char))

            # Собираем номер и выводим на основной кадр
            if detected_chars:
                detected_chars.sort(key=lambda x: x[0])
                final_number = ''.join([c[1] for c in detected_chars])
                cv2.putText(frame, final_number, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Отображение результата
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "C:/Users/alexanderdrozdov/Downloads/video2.MOV"  # Укажите правильный путь
    process_video(video_path)