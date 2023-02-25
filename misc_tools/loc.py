import glob


def stats(name, globbers):
    files = []
    for globber in globbers:
        files.extend(glob.glob(globber, recursive=True))
    lines = 0
    for file in files:
        with open(file) as fi:
            lines += len(fi.readlines())
    print(name, lines, sep='\t')


stats('main', ['torch/**/*.py', 'torch_redstone/**/*.py'])
stats('tests', ['tests/**/*.py'])
