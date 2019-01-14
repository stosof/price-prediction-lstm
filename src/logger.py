import logging


logger = logging.getLogger('General Logger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('../output/price-prediction-lstm.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def log_info(msg):
    print(msg)
    logger.info(msg)
