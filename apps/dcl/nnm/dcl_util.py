#
import datetime

class DclUtil(object):
    @staticmethod
    def datetime_format():
        return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")