from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER") 
# so now this registry is just a more fancy way of maintaining a dictionary in this case
#it has a name, and a set of key value pairs


def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    #so what the above line does is check if this TRAINER.NAME which we set in the configs
    #is present in the available trainers. If not, a value error would be thrown, else we continue on
    if cfg.VERBOSE:
        #verbose is basically if we want stuff to be displayed
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)



