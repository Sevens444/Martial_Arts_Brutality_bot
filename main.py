import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pyautogui
import time
import keyboard
import cv2
import cv2_ext
from numba import njit
pyautogui.PAUSE = 0.003
pyautogui.MINIMUM_DURATION = 0.009

limit_start_rad =  (    55,    130)
limit_start_area = (13_000, 50_000)
limit_arrow_area = (   300,  5_000)
limit_length_beetw_points = (11, 550)
block_height = 150
block_cor_coef = 1.22  # для диагональных блоков контрудар будет справа
dir_save = 'Fights_logs\\'

params = {
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


class Fight_master:
    def __init__(self, stage):
        self.stage = stage
        print(params[stage]['phrase'], end='\t')
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

        self.screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        self.screen_orig = self.screen.copy()
        self.cropped_image = self.screen[
                             params[self.stage]['limits_window']['y'][0]:params[self.stage]['limits_window']['y'][1],
                             params[self.stage]['limits_window']['x'][0]:params[self.stage]['limits_window']['x'][1]]

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
            if ((limit_start_rad[0] < radius < limit_start_rad[1]) and
                    (limit_start_area[0] < area < limit_start_area[1]) and
                    (self.gray_img[y][x] == 255) and
                    (sum([satur < 10 for satur in self.cropped_image[y][x]]) == 3)):
                self.atk_circle_center = (x + params[self.stage]['limits_window']['x'][0],
                                          y + params[self.stage]['limits_window']['y'][0])
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
            if limit_arrow_area[0] < area < limit_arrow_area[1]:
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
                        self.arrow_pos.append((cX + params[self.stage]['limits_window']['x'][0],
                                               cY + params[self.stage]['limits_window']['y'][0]))
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
                         and limit_length_beetw_points[0] < l < limit_length_beetw_points[1]]
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

            if ( x_new[-1] < params[self.stage]['limits_window']['x'][0] or
                 x_new[-1] > params[self.stage]['limits_window']['x'][1] or
                 y_new[-1] < params[self.stage]['limits_window']['y'][0] or
                 y_new[-1] > params[self.stage]['limits_window']['y'][1]):
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

        l = min(l) + params['defence']['limits_window']['x'][0]
        r = max(r) + params['defence']['limits_window']['x'][0]
        u = min(u) + params['defence']['limits_window']['y'][0]
        d = max(d) + params['defence']['limits_window']['y'][0]

        self.def_center = ((l + r) // 2, (u + d) // 2)

        # Нахождение точки контрудара
        # определение ориентации участка для блока
        if (r - l) < (d - u) * block_cor_coef:
            # Вертикальная ориентация. Смещение по x
            def_point_resist = (self.def_center[0] + block_height, self.def_center[1])
            def_point_half = (self.def_center[0] + int(block_height / 2), self.def_center[1])
            def_point_out = (self.def_center[0] - int(block_height / 2), self.def_center[1])
        else:
            # Горизонтальная ориентация. Смещение по y
            def_point_resist = (self.def_center[0], self.def_center[1] - block_height)
            def_point_half = (self.def_center[0], self.def_center[1] - int(block_height / 2))
            def_point_out = (self.def_center[0], self.def_center[1] + int(block_height / 2))

        self.def_follow_points = [def_point_resist, def_point_half, self.def_center, def_point_out]

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
            cv2_ext.imwrite(f"{dir_save}{self.dt_end_str}_{self.error}_orig.jpg", self.screen_orig)
        else:
            print(f"Выполнено за {self.dt_end - self.dt_start:.3f} секунд")
        return

def research_stages(fight_master):
    print('Функция сохранения этапов')
    # screen orig
    cv2_ext.imwrite(f"{dir_save}{fight_master.dt_end_str}_0_orig.jpg", fight_master.screen_orig)
    # gray img
    cv2_ext.imwrite(f"{dir_save}{fight_master.dt_end_str}_1_gray.jpg", fight_master.gray_img)

    if fight_master.stage == 'attack':
        print('# атака: вывод контуров начала удара')
        screen = fight_master.screen_orig.copy()
        for contour in fight_master.contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            area = cv2.contourArea(contour)
            # Проверка
            if ((limit_start_rad[0] < radius < limit_start_rad[1]) and
                    (limit_start_area[0] < area < limit_start_area[1]) and
                    (fight_master.gray_img[y][x] == 255) and
                    (sum([satur < 10 for satur in fight_master.cropped_image[y][x]]) == 3)):
                print((x, y), int(radius), int(area), fight_master.cropped_image[y][x], sep='\t')
                cont = [[[point[0][0] + params['attack']['limits_window']['x'][0],
                          point[0][1] + params['attack']['limits_window']['y'][0]
                          ]] for point in contour]
                cv2.drawContours(screen, np.array(cont), -1, (255, 0, 0), 3)
        cv2_ext.imwrite(f"{dir_save}{fight_master.dt_end_str}_2_contours.jpg", screen)

        print('# атака: вывод контуров стрелок и точек углов')
        screen = fight_master.screen_orig.copy()
        for contour in fight_master.contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            area = cv2.contourArea(contour)
            # Проверка
            if limit_arrow_area[0] < area < limit_arrow_area[1]:
                approx_angle = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                if ((3 <= len(approx_angle) <= 12) and
                        (fight_master.gray_img[y][x] == 0) and
                        (sum([(satur[0] < 10, satur[1] < 10, 170 < satur[2] < 195) for satur in [
                            fight_master.cropped_image[y][x]]][0]) == 3)):
                    print((x, y), int(radius), int(area), len(approx_angle), fight_master.cropped_image[y][x], sep='\t')
                    cont = [[[point[0][0] + params['attack']['limits_window']['x'][0],
                              point[0][1] + params['attack']['limits_window']['y'][0]
                              ]] for point in contour]
                    cv2.drawContours(screen, np.array(cont), -1, (255, 0, 0), 3)

                    for angle_point in approx_angle:
                        cont = [(point[0] + params['attack']['limits_window']['x'][0],
                                 point[1] + params['attack']['limits_window']['y'][0]
                                 ) for point in angle_point][0]
                        cv2.circle(screen, cont, 4, (0, 255, 0), thickness=2)
        cv2_ext.imwrite(f"{dir_save}{fight_master.dt_end_str}_3_circles.jpg", screen)

        if fight_master.follow_points_interpol:
            print('# атака: вывод точек траектории')
            screen = fight_master.screen_orig.copy()
            cv2.circle(screen, fight_master.follow_points_interpol[0], 10,(255, 0, 0), thickness=2)
            for point in fight_master.follow_points_interpol[1:]:
                cv2.circle(screen, point, 10, (0, 255, 0), thickness=2)
            for point in fight_master.arrow_pos:
                cv2.circle(screen, point, 10, (200, 200, 0), thickness=2)
            cv2_ext.imwrite(f"{dir_save}{fight_master.dt_end_str}_4_points.jpg", screen)

    if fight_master.stage == 'defence':
        screen = fight_master.screen_orig.copy()

        print('# защита: вывод контуров')
        for contour in fight_master.def_contours:
            cont = [[[point[0][0] + params['defence']['limits_window']['x'][0],
                      point[0][1] + params['defence']['limits_window']['y'][0]
                      ]] for point in contour]
            cv2.drawContours(screen, np.array(cont), -1, (255, 0, 0), 3)

        print('# защита: вывод точек траектории')
        for point in fight_master.def_follow_points:
            cv2.circle(screen, point, 4, (0, 0, 255), thickness=2)

        cv2_ext.imwrite(f"{dir_save}{fight_master.dt_end_str}_2_def.jpg", screen)

    print('###')

#%%
print('Битва начинается.')

# Основной цикл
while not keyboard.is_pressed('.') and not keyboard.is_pressed('/'):
    # Показать где должен быть левый верхний угол окна игры
    if keyboard.is_pressed('1'):
        pyautogui.moveTo((645,192))

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

    #   /                     /
    #  /  Стадия сохранения  /
    # /                    /
    if keyboard.is_pressed('s'):
        research_stages(fight_master)
        time.sleep(0.1)

    #
    time.sleep(0.1)
#
#
#