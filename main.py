import cv2
import numpy as np

def main():
    # 1. Загружаем муху с альфа-каналом (прозрачностью)
    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    
    if fly is None:
        print("Ошибка: не удалось загрузить fly64.png")
        print("Убедитесь, что файл находится в папке с программой")
        return
    
    # 2. Загружаем и преобразуем метку (по заданию 7 варианта)
    marker_img = cv2.imread('variant-7.jpg')
    
    if marker_img is None:
        print("Ошибка: не удалось загрузить изображение метки")
        return
    
    # Отражаем метку по заданию
    marker_transformed = cv2.flip(marker_img, -1)
    cv2.imwrite('marker_transformed.jpg', marker_transformed)
    print("Изображение метки преобразовано (отражение по обеим осям)")
    
    # 3. Открываем камеру
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    # Параметры для поиска желтой метки
    lower_color = np.array([15, 80, 80])
    upper_color = np.array([40, 255, 255])
    
    print("Нажмите 'q' для выхода")
    print("Дополнительное задание: муха накладывается на метку")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Рисуем центр кадра
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
        
        # Поиск метки (как в основном задании)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Берем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 100:  # Фильтр по площади
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    # Координаты центра метки
                    marker_x = int(M["m10"] / M["m00"])
                    marker_y = int(M["m01"] / M["m00"])
                    
                    # Рисуем контур метки
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (marker_x, marker_y), 5, (0, 255, 0), -1)
                    
                    # # ==========================================
                    # # ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: НАЛОЖЕНИЕ МУХИ
                    # # ==========================================
                    
                    # Получаем размеры мухи
                    fly_h, fly_w = fly.shape[:2]
                    
                    # Вычисляем координаты для вставки (центр мухи = центр метки)
                    top_left_x = marker_x - fly_w // 2
                    top_left_y = marker_y - fly_h // 2
                    
                    # Проверяем, чтобы муха не выходила за границы кадра
                    if (top_left_x >= 0 and top_left_y >= 0 and 
                        top_left_x + fly_w <= width and top_left_y + fly_h <= height):
                        
                        # Если муха с прозрачностью (4 канала: B, G, R, Alpha)
                        if fly.shape[2] == 4:
                            # Разделяем на цвет и альфа-канал
                            fly_bgr = fly[:, :, :3]
                            fly_alpha = fly[:, :, 3] / 255.0
                            
                            # Вырезаем область под мухой
                            roi = frame[top_left_y:top_left_y+fly_h, top_left_x:top_left_x+fly_w]
                            
                            # Накладываем с учетом прозрачности
                            for c in range(3):
                                roi[:, :, c] = (roi[:, :, c] * (1 - fly_alpha) + 
                                               fly_bgr[:, :, c] * fly_alpha)
                        else:
                            # Если муха без прозрачности - просто вставляем
                            frame[top_left_y:top_left_y+fly_h, top_left_x:top_left_x+fly_w] = fly
                        
                        # Рисуем рамку вокруг мухи
                        cv2.rectangle(frame, (top_left_x, top_left_y), 
                                    (top_left_x + fly_w, top_left_y + fly_h), (255, 0, 255), 1)
                    
                    # Вычисляем расстояние до центра (основное задание)
                    distance = np.sqrt((marker_x - center_x)**2 + (marker_y - center_y)**2)
                    
                    # Выводим информацию
                    cv2.putText(frame, f"Distance: {distance:.1f} px", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Fly at marker center", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Линия от метки до центра
                    cv2.line(frame, (marker_x, marker_y), (center_x, center_y), (255, 0, 255), 2)
                else:
                    cv2.putText(frame, "Marker found", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Lab8 + Fly Overlay', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
