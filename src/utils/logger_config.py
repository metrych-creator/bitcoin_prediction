import logging
import os

def get_logger(name="bitcoin_prediction"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # no logger duplicates
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file handler
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(f"logs/app.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    IS_LOCAL = os.getenv("IS_LOCAL", "True").lower() == "true"

    if IS_LOCAL:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("Running with Local Logger")
    else:
        # GOOGLE CLOUD LOGGER (Placeholder)
        try:
            # from google.cloud import logging as gcloud_logging
            # client = gcloud_logging.Client()
            # handler = client.get_default_handler()
            logger.info("Running with Google Cloud Logger [Placeholder]")
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        except Exception:
            logger.addHandler(logging.StreamHandler())

    return logger

logger = get_logger()