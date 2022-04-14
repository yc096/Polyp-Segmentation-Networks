import xlwt

class excel_logger():
    def __init__(self):
        self.work_book = xlwt.Workbook(encoding='UTF-8')

    def add_train_sheet(self,sheet_name):
        worksheet = self.work_book.add_sheet(sheet_name)
        worksheet.write(0, 0, 'Epoch')
        worksheet.write(0, 1, 'loss')
        worksheet.write(0, 2, 'lr')
        worksheet.write(0, 3, 'Time')
        return worksheet

    def add_test_sheet(self,sheet_name):
        worksheet = self.work_book.add_sheet(sheet_name)
        worksheet.write(0, 0, 'Epoch')
        worksheet.write(0, 1, 'loss')
        worksheet.write(0, 2, 'IoU')
        worksheet.write(0, 3, 'Dice')
        worksheet.write(0, 4, 'F1-score')
        worksheet.write(0, 5, 'MAE')
        worksheet.write(0, 6, 'accuracy')
        worksheet.write(0, 7, 'precision')
        worksheet.write(0, 8, 'recall')
        worksheet.write(0, 9, 'Time')
        return worksheet
    
    def save(self, path):
        self.work_book.save(filename_or_stream=path)
