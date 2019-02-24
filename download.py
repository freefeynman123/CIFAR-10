from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
import re
import tarfile

if __name__ == "__main__":
	html_body = urlopen("https://www.cs.toronto.edu/~kriz/cifar.html")
	soup = BeautifulSoup(html_body, 'html.parser')
	file_to_download = soup.find(href = re.compile('^.*python.tar.gz$'))
	file = file_to_download.get('href')
	# Omit cifar.html from original address and replace it by href found in file_to_download
	url_to_download = '/'.join(html_body.geturl().split('/')[:-1] + [file])
	urlretrieve(url_to_download, "cifar_data.tar.gz")
	tar = tarfile.open('cifar_data.tar.gz', 'r:gz')
	tar.extractall()
	tar.close()