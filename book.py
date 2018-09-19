
from fpdf import FPDF

title = 'ArtiCanon'

class PDF(FPDF):
    def book_cover(self):
        self.add_font('EB Garamond', '', r"/Users/usa_sun_shine/Library/Fonts/EBGaramond08-Regular.ttf", uni=True)
        self.set_font('EB Garamond', '', 30)
        self.set_xy(75, 20)
        self.cell(40, 10, "The Articanon")
        self.set_font('EB Garamond', '', 16)
        self.set_xy(87, 30)
        self.cell(40, 10, "by 209.51.170.10")
        self.image("/Users/usa_sun_shine/Downloads/articanon_cover1.jpg", x = 50.5, y = 60, w = 110, h = 180, type = '', link = '')


    def chapter_title(self, num, label):
        self.add_font('EB Garamond', '', r"/Users/usa_sun_shine/Library/Fonts/EBGaramond08-Regular.ttf", uni=True)
        self.set_font('EB Garamond', '', 15)
        self.set_fill_color(255, 255, 255)
        self.cell(1)
        self.cell(20, 10, 'Chapter %d' % (num), 0, 1, 'C', 1)
        self.set_font('EB Garamond', '', 30)
        self.cell(40)
        self.cell(30, 20, '%s' % (label), 0, 1, 'C', 1)
        self.ln(8)
        self.line(51, 50, 158.5, 50)

    def chapter_body(self, name):
        with open(name, 'rb') as fh:
            txt = fh.read().decode("UTF-8")
        self.set_font('EB Garamond', '', 12)
        self.multi_cell(0, 5, txt)
        self.ln()
        self.set_font('EB Garamond', '')
        self.cell(0, 5, '(end of chapter)')

    def print_chapter(self, num, title, name):
        self.set_margins(50,16,50)
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(name)

    def print_book_cover(self):
        self.add_page()
        self.book_cover()

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.cell(0, 10, ' %s' % self.page_no(), 0, 0, 'C')


def create_pdf(chapter_list):
    pdf = PDF()
    pdf.set_title(title)
    pdf.set_author('CavML')
    pdf.print_book_cover()
    chap_num = 1
    title_of_chap = "Wisdom"
    for chapter in chapter_list:
        pdf.print_chapter(chap_num, title_of_chap, chapter)
        chap_num += 1
    pdf.output('tuto3.pdf', 'F')

create_pdf(["full_text.txt"])
