# pdf_merger.py
import glob
from PyPDF2 import PdfFileWriter, PdfFileReader
def merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()
    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)

paths = glob.glob('pdf/*.pdf')
paths.sort()
merger('Merge.pdf', paths)

# import PyPDF2
#
# pdf_in = open('pdf/roomii.pdf', 'rb')
# pdf_reader = PyPDF2.PdfFileReader(pdf_in)
# pdf_writer = PyPDF2.PdfFileWriter()
# page = pdf_reader.getPage(0)
# page.rotateClockwise(90)
# pdf_writer.addPage(page)
# # for pagenum in range(pdf_reader.numPages):
# #     page = pdf_reader.getPage(pagenum)
# #     if pagenum % 2:
# #         page.rotateClockwise(90)
# #     pdf_writer.addPage(page)
#
# pdf_out = open('pdf/roomi.pdf', 'wb')
# pdf_writer.write(pdf_out)
# pdf_out.close()
# pdf_in.close()