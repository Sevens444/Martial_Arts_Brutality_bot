import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pyautogui
import time
import keyboard
import cv2
import cv2_ext
from numba import njit
import pygetwindow as gw
import ctypes
from ctypes import wintypes

# Определяем константы Windows API
HWND = wintypes.HWND
HDC = wintypes.HDC
HBITMAP = wintypes.HANDLE
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
UINT = wintypes.UINT
LPCWSTR = wintypes.LPWSTR
LPVOID = wintypes.LPVOID

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]

# Определяем структуру RGBTRIPLE
class RGBTRIPLE(ctypes.Structure):
    _fields_ = [
        ("rgbtBlue", ctypes.c_ubyte),
        ("rgbtGreen", ctypes.c_ubyte),
        ("rgbtRed", ctypes.c_ubyte),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", RGBTRIPLE * 1)  # Используем нашу структуру RGBTRIPLE
    ]

# Загружаем необходимые DLL
gdi32 = ctypes.windll.gdi32
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

#  функции Windows API
GetWindowDC = user32.GetWindowDC
ReleaseDC = user32.ReleaseDC
CreateCompatibleDC = gdi32.CreateCompatibleDC
CreateCompatibleBitmap = gdi32.CreateCompatibleBitmap
SelectObject = gdi32.SelectObject
BitBlt = gdi32.BitBlt
GetDIBits = gdi32.GetDIBits
DeleteObject = gdi32.DeleteObject
GetDeviceCaps = gdi32.GetDeviceCaps


def screenshot_to_numpy():
    """
    Делает скриншот окна и возвращает numpy array.

    :return numpy.ndarray: numpy array с изображением в формате RGB (height, width, 3) или None в случае ошибки.
    """
    # window_name = "название_окна"
    # hwnd = gw.getWindowsWithTitle(window_name)[0]._hWnd  # Получаем HWND окна по имени
    hwnd = None  # получить скриншот по названию окна или None - всего экрана

    try:
        if hwnd is None:
            hwnd = user32.GetDesktopWindow()

        # Получаем дескриптор контекста устройства (HDC)
        hdc = GetWindowDC(hwnd)

        # Получаем размеры окна
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width = rect.right - rect.left
        height = rect.bottom - rect.top

        # Создаем совместимый DC
        memdc = CreateCompatibleDC(hdc)

        # Создаем совместимую растровую карту
        bitmap = CreateCompatibleBitmap(hdc, width, height)

        # Выбираем растровую карту в DC
        SelectObject(memdc, bitmap)

        # Копируем содержимое окна в совместимый DC
        BitBlt(memdc, 0, 0, width, height, hdc, 0, 0, SRCCOPY)

        # Создаем структуру BITMAPINFO
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 24
        bmi.bmiHeader.biCompression = BI_RGB

        # Выделяем память под пиксели
        buffer_size = width * height * 3
        buffer = (ctypes.c_ubyte * buffer_size)()

        # Получаем биты растровой карты
        GetDIBits(memdc, bitmap, 0, height, buffer, ctypes.byref(bmi), DIB_RGB_COLORS)

        # Конвертируем в numpy array
        image_array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))

        # Конвертируем BGR в RGB
        # image_array = image_array[:, :, ::-1]

        # Освобождаем ресурсы
        DeleteObject(bitmap)
        DeleteObject(memdc)
        ReleaseDC(hwnd, hdc)

        return image_array
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None


pyautogui.PAUSE = 0.003
pyautogui.MINIMUM_DURATION = 0.009

LIMIT_START_RAD =  (    55,    130)
LIMIT_START_AREA = (13_000, 50_000)
LIMIT_ARROW_AREA = (   300,  5_000)
LIMIT_LENGTH_BEETW_POINTS = (11, 550)
BLOCK_HEIGHT = 150
BLOCK_CORR_COEFF = 1.22  # для диагональных блоков контрудар будет справа
DIR_SAVE = 'C:\\Users\\Nikolay\\Documents\\Python Files\\Martial_Arts_Brutality_bot\\Fights_logs\\'

PARAMS = {
    'attack' : {
        'phrase': 'Беру удар на себя.',
        'limits_window' : {'x': (700, 1820),
                           'y': (300, 1180)}
    },
    'defence' : {
        'phrase': 'Беру защиту на себя.',
        'limits_window' : {'x': (700, 1850),
                           'y': (400, 1180)}
    },
    'counter_attack': {
        'phrase' : 'Беру контр атаку на себя.'
    }
}

@njit
def calculate_distances(last_pos, arrow_pos):
    return np.sqrt(np.sum((arrow_pos - last_pos) ** 2, axis=1))
# при первом запуске долго выполняется из-за numpy warnings возникающих при импорте в PyCharm
calculate_distances(np.array([819, 738]), np.array([[999,999],[888,888],[777,777],[666,666]]))

class Fight_master:
    def __init__(self, stage):
        print(PARAMS[stage]['phrase'], end='\t')
        self.stage = stage
        self.dt_start = time.time()
        self.error = None
        self.dt_end_str = time.strftime("%Y_%m_%d %H.%M.%S", time.localtime())

        if stage == 'attack':
            self.contours = None
            self.atk_circle_center = None
            self.arrow_pos = []
            self.already_used_ind = []
            self.follow_points = None
            self.follow_points_interpol = None

        if stage == 'defence':
            self.def_contours = None
            self.def_center = None
            self.def_point_resist = None
            self.def_point_half = None
            self.def_follow_points = []

        self.screen = screenshot_to_numpy()
        self.cropped_image = self.screen[
                             PARAMS[self.stage]['limits_window']['y'][0]:PARAMS[self.stage]['limits_window']['y'][1],
                             PARAMS[self.stage]['limits_window']['x'][0]:PARAMS[self.stage]['limits_window']['x'][1]]

    def atk_find_black_circle(self):
        '''
        Нахождение центра черного кружка - начало удара
        '''
        if self.screen is None:
            self.error = 'Не удалось получить скриншот'
            return

        gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)

        _, self.gray_img = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        self.contours, _ = cv2.findContours(self.gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in self.contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            area = cv2.contourArea(contour)

            # Проверка по радиусу, площади, цвету центра выделения (должен быть макс приближен к черному)
            if ((LIMIT_START_RAD[0] < radius < LIMIT_START_RAD[1]) and
                    (LIMIT_START_AREA[0] < area < LIMIT_START_AREA[1]) and
                    (self.gray_img[y][x] == 255) and
                    (sum([satur < 10 for satur in self.cropped_image[y][x]]) == 3)):
                self.atk_circle_center = (x + PARAMS[self.stage]['limits_window']['x'][0],
                                          y + PARAMS[self.stage]['limits_window']['y'][0])
                return

        self.error = 'Начало атк не найден'


    def atk_find_arrow_centers(self):
        '''
        Нахождение центров объектов похожих на стрелки
        '''
        if self.error:
            return

        for contour in self.contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            area = cv2.contourArea(contour)

            # Проверка по площади
            if LIMIT_ARROW_AREA[0] < area < LIMIT_ARROW_AREA[1]:
                approx_angle = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                # Проверка кол-ву углов, цвета центра выделения по изображению ргб и чб
                if ( (3 <= len(approx_angle) <= 12) and
                        (self.gray_img[y][x] == 0) and
                        (sum([(satur[0] < 10, satur[1] < 10, 170 < satur[2] < 195) for satur in [
                            self.cropped_image[y][x]]][0]) == 3)):

                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        self.arrow_pos.append((cX + PARAMS[self.stage]['limits_window']['x'][0],
                                               cY + PARAMS[self.stage]['limits_window']['y'][0]))
        self.arrow_pos = np.array(self.arrow_pos)
        return


    def atk_close_arrow_center(self):
        '''
        Нахождение оптимальной траектории удара
        Создание списка следующих друг за другом стрелок в установленных пределах
        '''
        last_pos = np.array(self.follow_points[-1])
        lengths = calculate_distances(last_pos, self.arrow_pos)

        valid_indices = [i for i, l in enumerate(lengths) if i not in self.already_used_ind
                         and LIMIT_LENGTH_BEETW_POINTS[0] < l < LIMIT_LENGTH_BEETW_POINTS[1]]

        if valid_indices:
            closest_index = valid_indices[np.argmin(lengths[valid_indices])]
            self.already_used_ind.append(closest_index)
            self.follow_points.append(tuple(self.arrow_pos[closest_index]))
            return True
        return False


    def atk_add_points(self, num_points_between=4, num_points_ahead=4):
        '''
        Добавление точек после найденных и расчет оптимальной траектории удара

        :param self.follow_points: Массив центров стрелок
        :param num_points_between: Кол-во добавляемых точек между найденными
        :param num_points_ahead: Кол-во добавляемых точек после найденных
        :return: Массив точек интерполированной (сглаженной) оптимальной траектории удара
        '''
        # Продолжение удара
        x, y = zip(*self.follow_points)
        x = list(x)
        y = list(y)

        for i in range(1, num_points_ahead + 1):
            x.append(int(4 * x[-1] - 6 * x[-2] + 4 * x[-3] - x[-4]))
            y.append(int(4 * y[-1] - 6 * y[-2] + 4 * y[-3] - y[-4]))

        # Интерполяция между существующими точками
        x_new = []
        y_new = []

        for i in range(len(x) - 1):
            dx = (x[i + 1] - x[i]) / num_points_between
            dy = (y[i + 1] - y[i]) / num_points_between
            x_new.extend([int(x[i] + dx * n) for n in range(num_points_between + 1)])
            y_new.extend([int(y[i] + dy * n) for n in range(num_points_between + 1)])

            if ( x_new[-1] < PARAMS[self.stage]['limits_window']['x'][0] or
                 x_new[-1] > PARAMS[self.stage]['limits_window']['x'][1] or
                 y_new[-1] < PARAMS[self.stage]['limits_window']['y'][0] or
                 y_new[-1] > PARAMS[self.stage]['limits_window']['y'][1]):
                self.follow_points_interpol = list(zip(x_new, y_new))
                return

        x_new.append(x[-1])
        y_new.append(y[-1])

        self.follow_points_interpol = list(zip(x_new, y_new))
        return


    def atk_create_mouse_way(self):
        if self.error:
            return

        self.follow_points = [self.atk_circle_center]
        while self.atk_close_arrow_center():
            pass

        if len(self.follow_points) == 1:
            self.error = 'Путь атк не определен'
        else:
            self.atk_add_points()
        return


    def defence_find_contours_n_breaker(self):
        '''
        Нахождение контура защиты
        '''
        if self.screen is None:
            self.error = 'Не удалось получить скриншот'
            return

        lower_green = np.array([0, 100, 0])
        upper_green = np.array([10, 255, 100])
        self.gray_img = cv2.inRange(self.cropped_image, lower_green, upper_green)
        self.def_contours, _ = cv2.findContours(self.gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not self.def_contours:
            self.error = 'Обл защ не найдена'
            return

        # нахождение крайней левой, правой, верхней, нижней точек
        p = self.def_contours[0][0][0]
        l, r, u, d = [p[0]], [p[0]], [p[1]], [p[1]]
        for def_contour in self.def_contours:
            if len(def_contour) > 10:
                points_x = [p[0][0] for p in def_contour]
                points_y = [p[0][1] for p in def_contour]

                l.append(min(points_x))
                r.append(max(points_x))
                u.append(min(points_y))
                d.append(max(points_y))

        l = min(l) + PARAMS['defence']['limits_window']['x'][0]
        r = max(r) + PARAMS['defence']['limits_window']['x'][0]
        u = min(u) + PARAMS['defence']['limits_window']['y'][0]
        d = max(d) + PARAMS['defence']['limits_window']['y'][0]

        self.def_center = ((l + r)//2, (u + d)//2)

        # Нахождение точки контрудара
        # определение ориентации участка для блока
        if (r - l) < (d - u) * BLOCK_CORR_COEFF:
            # Вертикальная ориентация. Смещение по x
            def_points_resist = [(self.def_center[0] + int(BLOCK_HEIGHT * (k / 10)), self.def_center[1]) for k in range(10,1,-1)]
            def_point_out = (self.def_center[0] - int(BLOCK_HEIGHT * 0.5), self.def_center[1])
        else:
            # Горизонтальная ориентация. Смещение по y
            def_points_resist = [(self.def_center[0], self.def_center[1] - int(BLOCK_HEIGHT * (k / 10))) for k in range(10,1,-1)]
            def_point_out = (self.def_center[0], self.def_center[1] + int(BLOCK_HEIGHT * 0.5))

        self.def_follow_points = [*def_points_resist, self.def_center, def_point_out]

        return
    

    def mouse_action(self):
        '''
        Авто управление мыши. Движение по найденной траектории
        '''
        if self.error:
            pyautogui.moveTo((1280, 800))
            return

        if self.stage == 'attack':
            point_start = self.atk_circle_center
            points_way = self.follow_points_interpol

        if self.stage == 'defence':
            point_start = self.def_follow_points[0]
            points_way = self.def_follow_points[1:]

        pyautogui.moveTo(point_start)
        time.sleep(0.014)
        pyautogui.mouseDown()
        for point in points_way:
            pyautogui.moveTo(point)
        pyautogui.mouseUp()

        pyautogui.moveTo((1280, 800))

        return

    def show_info_n_save_err_screen(self):
        # Конец обработки удара / защиты
        self.dt_end_str = time.strftime("%Y_%m_%d %H.%M.%S", time.localtime())
        self.dt_end = time.time()

        # Сохранение оригинала при ошибке
        if self.error:
            print(self.error)
            cv2_ext.imwrite(f"{DIR_SAVE}{self.dt_end_str}_{self.error}_orig.jpg", self.screen)
        else:
            print(f"Выполнено за {(self.dt_end - self.dt_start)*1000:.0f} мс")
        return

def research_stages(fight_master):
    print('Функция сохранения этапов')
    # screen orig
    cv2_ext.imwrite(f"{DIR_SAVE}{fight_master.dt_end_str}_0_orig.jpg", fight_master.screen)
    # gray img
    cv2_ext.imwrite(f"{DIR_SAVE}{fight_master.dt_end_str}_1_gray.jpg", fight_master.gray_img)

    if fight_master.stage == 'attack':
        print('# атака: вывод контуров начала удара')
        screen = fight_master.screen.copy()
        for contour in fight_master.contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            area = cv2.contourArea(contour)
            # Проверка
            if ((LIMIT_START_RAD[0] < radius < LIMIT_START_RAD[1]) and
                    (LIMIT_START_AREA[0] < area < LIMIT_START_AREA[1]) and
                    (fight_master.gray_img[y][x] == 255) and
                    (sum([satur < 10 for satur in fight_master.cropped_image[y][x]]) == 3)):
                print((x, y), int(radius), int(area), fight_master.cropped_image[y][x], sep='\t')
                cont = [[[point[0][0] + PARAMS['attack']['limits_window']['x'][0],
                          point[0][1] + PARAMS['attack']['limits_window']['y'][0]
                          ]] for point in contour]
                cv2.drawContours(screen, np.array(cont), -1, (255, 0, 0), 3)
        cv2_ext.imwrite(f"{DIR_SAVE}{fight_master.dt_end_str}_2_contours.jpg", screen)

        print('# атака: вывод контуров стрелок и точек углов')
        screen = fight_master.screen.copy()
        for contour in fight_master.contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            area = cv2.contourArea(contour)
            # Проверка
            if LIMIT_ARROW_AREA[0] < area < LIMIT_ARROW_AREA[1]:
                approx_angle = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                if ((3 <= len(approx_angle) <= 12) and
                        (fight_master.gray_img[y][x] == 0) and
                        (sum([(satur[0] < 10, satur[1] < 10, 170 < satur[2] < 195) for satur in [
                            fight_master.cropped_image[y][x]]][0]) == 3)):
                    print((x, y), int(radius), int(area), len(approx_angle), fight_master.cropped_image[y][x], sep='\t')
                    cont = [[[point[0][0] + PARAMS['attack']['limits_window']['x'][0],
                              point[0][1] + PARAMS['attack']['limits_window']['y'][0]
                              ]] for point in contour]
                    cv2.drawContours(screen, np.array(cont), -1, (255, 0, 0), 3)

                    for angle_point in approx_angle:
                        cont = [(point[0] + PARAMS['attack']['limits_window']['x'][0],
                                 point[1] + PARAMS['attack']['limits_window']['y'][0]
                                 ) for point in angle_point][0]
                        cv2.circle(screen, cont, 4, (0, 255, 0), thickness=2)
        cv2_ext.imwrite(f"{DIR_SAVE}{fight_master.dt_end_str}_3_circles.jpg", screen)

        if fight_master.follow_points_interpol:
            print('# атака: вывод точек траектории')
            screen = fight_master.screen.copy()
            cv2.circle(screen, fight_master.follow_points_interpol[0], 10,(255, 0, 0), thickness=2)
            for point in fight_master.follow_points_interpol[1:]:
                cv2.circle(screen, point, 10, (0, 255, 0), thickness=2)
            for point in fight_master.arrow_pos:
                cv2.circle(screen, point, 10, (200, 200, 0), thickness=2)
            cv2_ext.imwrite(f"{DIR_SAVE}{fight_master.dt_end_str}_4_points.jpg", screen)

    if fight_master.stage == 'defence':
        screen = fight_master.screen.copy()

        print('# защита: вывод контуров')
        for contour in fight_master.def_contours:
            cont = [[[point[0][0] + PARAMS['defence']['limits_window']['x'][0],
                      point[0][1] + PARAMS['defence']['limits_window']['y'][0]
                      ]] for point in contour]
            cv2.drawContours(screen, np.array(cont), -1, (255, 0, 0), 3)

        print('# защита: вывод точек траектории')
        for point in fight_master.def_follow_points:
            cv2.circle(screen, point, 4, (0, 0, 255), thickness=2)

        cv2_ext.imwrite(f"{DIR_SAVE}{fight_master.dt_end_str}_2_def.jpg", screen)

    print('###')

#%%
print('Битва начинается.')

# Основной цикл
while not keyboard.is_pressed('.') and not keyboard.is_pressed('/'):
    # показать где должен быть левый верхний угол окна игры
    if keyboard.is_pressed('1'):
        pyautogui.moveTo((645, 192))

    #   /                   /
    #  /  Стадия 1. Атака  /
    # /                   /
    if keyboard.is_pressed('z'):
        pyautogui.moveTo((645, 192))
        fight_master = Fight_master('attack')
        fight_master.atk_find_black_circle()
        fight_master.atk_find_arrow_centers()
        fight_master.atk_create_mouse_way()
        fight_master.mouse_action()
        fight_master.show_info_n_save_err_screen()
        time.sleep(0.1)

    #   /                    /
    #  /  Стадия 2. Защита  /
    # /                    /
    if keyboard.is_pressed('x'):
        fight_master = Fight_master('defence')
        fight_master.defence_find_contours_n_breaker()
        fight_master.mouse_action()
        fight_master.show_info_n_save_err_screen()
        time.sleep(0.1)

    #   /                    /
    #  /  Вызов сохранения  /
    # /                    /
    if keyboard.is_pressed('s'):
        research_stages(fight_master)
        time.sleep(0.1)

    #
    time.sleep(0.1)
#
#
#