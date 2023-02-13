import requests
import bs4
import tqdm

version_list = [
    '0.4.1',
    '1.0.0',
    '1.1.0',
    '1.2.0',
    '1.3.0',
    '1.4.0',
    '1.5.0',
    '1.6.0',
    '1.7.0',
    '1.8.0',
    '1.9.0',
    '1.10',
    '1.11',
    '1.12',
]
torch_uri = 'https://pytorch.org/docs/%s/torch.html'
tracked = set()


def try_track(api):
    if not api or api.startswith('('):
        return
    if api not in tracked:
        print(api, file=file)
        tracked.add(api)


with open('logs/docdiff.txt', 'w') as file:
    for version in tqdm.tqdm(version_list):
        doc = requests.get(torch_uri % version).text
        print("New in " + version, file=file)
        bs = bs4.BeautifulSoup(doc, features='html.parser')
        for element in bs.find_all('dt', recursive=True):
            idx = element.get('id')
            if idx and idx.startswith('torch.'):
                try_track(idx)
        for element in bs.find_all('a', recursive=True):
            if element.get('class') and 'reference' in element.get('class'):
                try_track(element.get('title'))
        print(file=file)
