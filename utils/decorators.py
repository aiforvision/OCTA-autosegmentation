def overrides(interface_class):
    """
    This method overrides a method from the given interface
    """
    def overrider(method):
        assert (method.__name__ in dir(interface_class)), f"Overriden method {method.__name__} does not exist in interface {interface_class.__name__}!"
        return method
    return overrider