CONFIG_MAPPING = {}


def _register(name, import_path, class_name='TrainConfig'):
    try:
        module = __import__(import_path, fromlist=[class_name])
        CONFIG_MAPPING[name] = getattr(module, class_name)
    except ModuleNotFoundError:
        pass


_register('aic_cable_insertion', 'experiments.aic_cable_insertion.config')
_register('ram_insertion', 'experiments.ram_insertion.config')
_register('usb_pickup_insertion', 'experiments.usb_pickup_insertion.config')
_register('object_handover', 'experiments.object_handover.config')
_register('egg_flip', 'experiments.egg_flip.config')
