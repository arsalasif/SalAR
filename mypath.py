from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def save_root_dir():
        return './models'

    @staticmethod
    def models_dir():
        return "./models"

