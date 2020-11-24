from os.path import isdir
from pathlib import Path
import glob
import os
import shutil

class FileSystem():
    def __init__(self, use_localpath=False):
        def process_path(path):
            import os
            if type(path) != str:
                return path
            if os.path.isabs(path):
                return path
            if use_localpath:
                return self.relative_to_caller(path, go_up=1)
            else:
                return os.path.abspath(path)
                
        self.process_path = process_path
        
        # aliases
        self.is_folder = self.is_directory
        self.is_dir    = self.is_directory
        self.glob      = glob.glob
        
    def write(self, data, to=None):
        to = self.process_path(to)
        # make sure the path exists
        self.makedirs(os.path.dirname(to))
        with open(to, 'w') as the_file:
            the_file.write(str(data))
    
    def read(self, filepath):
        filepath = self.process_path(filepath)
        try:
            with open(filepath,'r') as f:
                output = f.read()
        except:
            output = None
        return output    
        
    def delete(self, filepath):
        filepath = self.process_path(filepath)
        if isdir(filepath):
            shutil.rmtree(filepath)
        else:
            try:
                os.remove(filepath)
            except:
                pass
    
    def makedirs(self, path):
        path = self.process_path(path)
        try:
            os.makedirs(path)
        except:
            pass
        
    def copy(self, original=None, to=None, new_name="", force=True):
        original = self.process_path(original)
        to = self.process_path(to)
        # failsafe for new name
        if new_name == "":
            raise Exception('self.copy() needs a new_name= argument:\n    self.copy(original="location", to="directory", new_name="")\nif you want the name to be the same as before do new_name=None')
        elif new_name is None:
            new_name = os.path.basename(original)
        
        # get the full path
        to = os.path.join(to, new_name)
        # if theres a file in the target, delete it
        if force and self.exists(to):
            self.delete(to)
        # make sure the containing folder exists
        self.makedirs(os.path.dirname(to))
        if os.path.isdir(original):
            shutil.copytree(original, to)
        else:
            return shutil.copy(original, to)
    
    def move(self, original=None, to=None, new_name="", force= True):
        original = self.process_path(original)
        to = self.process_path(to)
        if new_name == "":
            raise Exception('self.move() needs a new_name= argument:\n    self.move(original="location", to="directory", new_name="")\nif you want the name to be the same as before do new_name=None')
        elif new_name is None:
            new_name = os.path.basename(original)
        
        # get the full path
        to = os.path.join(to, new_name)
        # make sure the containing folder exists
        self.makedirs(os.path.dirname(to))
        shutil.move(original, to)
    
    def exists(self, path):
        path = self.process_path(path)
        return os.path.exists(path)
        
    def is_directory(self, path):
        path = self.process_path(path)
        return os.path.isdir(path)
    
    def is_file(self, path):
        path = self.process_path(path)
        return os.path.isfile(path)

    def list_files(self, path="."):
        path = self.process_path(path)
        return [ x for x in self.ls(path) if self.is_file(x) ]
    
    def list_folders(self, path="."):
        path = self.process_path(path)
        return [ x for x in self.ls(path) if self.is_folder(x) ]
    
    def glob(self, *args, **kwargs):
        return glob.glob(*args, **kwargs)
        
    def ls(self, filepath="."):
        filepath = self.process_path(filepath)
        glob_val = filepath
        if os.path.isdir(filepath):
            glob_val = os.path.join(filepath, "*")
        return glob.glob(glob_val)

    def touch(self, path):
        path = self.process_path(path)
        self.makedirs(self.dirname(path))
        if not self.exists(path):
            self.write("", to=path)
    
    def touch_dir(self, path):
        path = self.process_path(path)
        self.makedirs(path)
    
    def dirname(self, path):
        path = self.process_path(path)
        return os.path.dirname(path)
    
    def basename(self, path):
        path = self.process_path(path)
        return os.path.basename(path)
    
    def extname(self, path):
        path = self.process_path(path)
        filename, file_extension = os.path.splitext(path)
        return file_extension
    
    def path_pieces(self, path):
        """
        example:
            *folders, file_name, file_extension = self.path_pieces("/this/is/a/filepath.txt")
        """
        path = self.process_path(path)
        folders = []
        while 1:
            path, folder = os.path.split(path)

            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)

                break
        folders.reverse()
        *folders, file = folders
        filename, file_extension = os.path.splitext(file)
        return [ *folders, filename, file_extension ]
    
    def join(self, *paths):
        return os.path.join(*paths)
    
    def relative_to_caller(self, path, go_up=0):
        import os
        import inspect
        the_stack = inspect.stack()[2+go_up]
        the_module = inspect.getmodule(the_stack[0])
        if the_module == None:
            parent_path = os.getcwd()
        else:
            parent_path = os.path.dirname(the_module.__file__)
        return os.path.join(parent_path, path)
        
    def make_absolute_path(self, path):
        path = self.process_path(path)
        return os.path.abspath(path)

    def isabsolute_path(self, path):
        return os.path.isabs(self.process_path(path))

    def pwd(self):
        return os.getcwd()
    
    def size(self, path):
        """
        returns size in bytes of either a folder or a file
        """
        path = self.process_path(path)
        if self.is_file(path):
            return os.path.getsize(path)
        else:
            total_size = 0
            for each_path in Path(path).glob('**/*'):
                total_size += os.path.getsize(each_path)
        
        return total_size

FS = FileSystem()
FSL = FileSystem(use_localpath=True)