from bbrl import instantiate_class

class Logger:
    """
    A class for logging data using a specified logger instance.

    Parameters:
    - cfg (dict): Configuration dictionary containing logger settings.
    """

    def __init__(self, cfg):
        """
        Initialize the Logger instance.

        Args:
        - cfg (dict): Configuration dictionary containing logger settings.
        """
        self.logger = instantiate_class(cfg.logger)
        
    def add_log(self, log_string, data, steps):
        """
        Add a log entry to the logger.

        Args:
        - log_string (str): The name or identifier of the log entry.
        - data (torch.Tensor or float): The data to be logged.
        - steps (int): The step or iteration number associated with the log entry.
        """
        self.logger.add_scalar(log_string, data.item(), steps)
