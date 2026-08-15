idf_component_register(SRCS "app_main.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp-nn
                       PRIV_REQUIRES esp_timer esp_driver_gpio)
