import requests
from bs4 import BeautifulSoup
import io
from PyPDF2 import PdfFileReader


url = "https://www.bombeiros.go.gov.br/sem-categoria/normas-tecnicas-do-cbmgo-2.html"
read = requests.get(url)
html_content = read.content
soup = BeautifulSoup(html_content, "html.parser")

list_of_pdf = set()
element = soup.find_all('p')
p = element[1].find_all('a')

for link in (p):
    pdf_link = ("https://www.bombeiros.go.gov.br/" + link.get('href'))
    print(pdf_link)
    list_of_pdf.add(pdf_link)


def info(pdf_path):
    response = requests.get(pdf_path)

    with io.BytesIO(response.content) as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()
        content = pdf.getPage(3).extractText()

    txt = f"""
    Information about {pdf_path}:
    Author: {information.author}
    Creator: {information.creator}
    Producer: {information.producer}
    Subject: {information.subject}
    Title: {information.title}
    Number of pages: {number_of_pages}
    Content: {content}
    """
    print(txt)
    return information


for i in list_of_pdf:
    info(i)
