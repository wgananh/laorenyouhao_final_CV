import time


class Time_Controller:
    def __init__(self):
        # 数据库插入的计时器
        self.insert_timing = 0  # 计时开始
        self.insert_start_time = 0  # 开始时间
        self.insert_limit_time = 60

        # 控制禁止区域监测
        self.fence_timing = 0  # 计时开始
        self.fence_start_time = 0  # 开始时间
        self.fence_limit_time = 10

        # 控制陌生人检测
        self.strangers_timing = 0  # 计时开始
        self.strangers_start_time = 0  # 开始时间
        self.strangers_limit_time = 10

        # 控制微笑检测
        self.facial_expression_timing = 0  # 计时开始
        self.facial_expression_start_time = 0  # 开始时间
        self.facial_expression_limit_time = 2

        # 控制跌倒检测
        self.fall_timing = 0  # 计时开始
        self.fall_start_time = 0  # 开始时间
        self.fall_limit_time = 3

        # 控制义工交互检测
        self.activity_timing = 0
        self.activity_start_time = 0
        self.activity_limit_time = 5

    def set_insert_timing(self, insert_timing):
        self.insert_timing = insert_timing

    def set_fence_timing(self, fence_timing):
        self.fence_timing = fence_timing

    def set_stranger_timing(self, strangers_timing):
        self.strangers_timing = strangers_timing

    def set_facial_expression_timing(self, facial_expression_timing):
        self.facial_expression_timing = facial_expression_timing

    def set_fall_timing(self, fall_timing):
        self.fall_timing = fall_timing

    def set_activity_timing(self, activity_timing):
        self.activity_timing = activity_timing

    def start_insert_timing(self):
        self.insert_start_time = time.time()

    def start_fence_timing(self):
        self.fence_start_time = time.time()

    def start_stranger_time(self):
        self.strangers_start_time = time.time()

    def start_facial_expression_time(self):
        self.facial_expression_start_time = time.time()

    def start_fall_start_time(self):
        self.fall_start_time = time.time()

    def start_activity_start_time(self):
        self.activity_start_time = time.time()
