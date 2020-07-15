import sys, os, hashlib, json, traceback, fnmatch, datetime, pathlib
import collections



def enum(sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)



def DoesPathExistAndIsDirectory(pathStr):
    if os.path.exists(pathStr) and os.path.isdir(pathStr):
        return True
    else:
        return False



def EnsureDirectoryExists(pathStr):
    if not DoesPathExistAndIsDirectory(pathStr):
        try:
            # os.mkdir(pathStr)
            pathlib.Path(pathStr).mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            err_fname = './errors.log'
            exc_type, exc_value, exc_traceback = sys.exc_info()
            with open(err_fname, 'a') as errf:
                traceback.print_tb(exc_traceback, limit=None, file=errf)
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=errf)
            print(str(ex))
            print('the directory you are trying to place a file to doesn\'t exist and cannot be created:\n%s' % pathStr)
            raise FileNotFoundError('the directory you are trying to place a file to doesn\'t exist and cannot be created:')



def find_files(directory, pattern, maxdepth=None):
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\\\', os.sep)
                if maxdepth is None:
                    flist.append(filename)
                else:
                    if filename.count(os.sep)-directory.count(os.sep) <= maxdepth:
                        flist.append(filename)
    return flist


def find_directories(directory, pattern=None, maxdepth=None):
    dirlist = []
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if pattern is None:
                retname = os.path.join(root, d, '')
                yield retname
            elif fnmatch.fnmatch(d, pattern):
                retname = os.path.join(root, d, '')
                retname = retname.replace('\\\\', os.sep)
                if maxdepth is None:
                    dirlist.append(retname)
                else:
                    if retname.count(os.sep)-directory.count(os.sep) <= maxdepth:
                        dirlist.append(retname)
    return dirlist




def dict_hash(dict_to_hash):
    hashvalue = hashlib.sha256(json.dumps(dict_to_hash, sort_keys=True).encode())
    return hashvalue.hexdigest()



def DoesPathExistAndIsFile(pathStr):
    if os.path.exists(pathStr) and os.path.isfile(pathStr):
        return True
    else:
        return False


def prime_factors(n):
    """Returns all the prime factors of a positive integer"""
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n /= d
        d = d + 1

    return factors


def uniques(items):
    unique = []
    for value in items:
        if value not in unique:
            unique.append(value)
    return unique



def ReportException(err_fname, ex):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    with open(err_fname, 'a') as errf:
        errf.write('================ ' + str(datetime.datetime.now()) + ' ================\n')
        traceback.print_tb(exc_traceback, limit=None, file=errf)
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=errf)
        errf.write('\n\n\n')



def isSequence(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.Sequence)