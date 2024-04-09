from bbrl import instantiate_class

class Logger():

    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, data, steps):
        self.logger.add_scalar(log_string, data.item(), steps)
