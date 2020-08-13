# import openpyxl
# from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
# import openpyxl.styles as sty
# # from openpyxl import Workbook ,load_workbook
# # import json
# #
# # fileName = "../data"
# # wb=load_workbook(fileName+".xlsx")
# # ws = wb["Sheet1"]
# # list_key=[]
# # jsonLine=[]
# # c = 8
# # r = 4140
# # for col in range(1,c+1):
# #     list_key.append(ws.cell(row=1,column=col).value)
# # for row in range(2,r+1):
# #     dict_v={}
# #     for col in range(1,c+1):
# #         dict_v[list_key[col-1]]=ws.cell(row=row,column=col).value
# #     jsonLine.append(dict_v)
# # json.dump(jsonLine,open("data"+".json","w",encoding="utf-8"))
#
# import json
# import openpyxl
# from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
# import openpyxl.styles as sty
# from openpyxl import Workbook ,load_workbook
#
# org_file = "/Users/xiayan/Desktop/对话式机器阅读理解/Multiparty-Dialog-RC-master/dialog_rc_data/json/Orginal.json"
# Trn_file = "/Users/xiayan/Desktop/对话式机器阅读理解/Multiparty-Dialog-RC-master/dialog_rc_data/json/Trn.json"
# Dev_file = "/Users/xiayan/Desktop/对话式机器阅读理解/Multiparty-Dialog-RC-master/dialog_rc_data/json/Dev.json"
# Tst_file = "/Users/xiayan/Desktop/对话式机器阅读理解/Multiparty-Dialog-RC-master/dialog_rc_data/json/Tst.json"
#
# org = open(org_file,'r')
# Trn = open(Trn_file,'w')
# Dev = open(Dev_file,'w')
# Tst = open(Tst_file,'w')
# # count = 0
# # for line in org:
# #
# #     jsonLine = json.loads(line)
# #     if count <= 5553:
# #         json.dump(jsonLine,Trn)
# #     elif count <= 7622:
# #         json.dump(jsonLine,Dev)
# #     else:
# #         json.dump(jsonLine,Tst)
# #     count += 1
#
# with open(Trn_file,'w') as Trn:
#     count = 0
#     samples = json.load(org)
#     for sample in samples:
#         if count <= 5553:
#
#             json.dump(sample,Trn)
#             Trn.write(', ')
#         elif count <= 6722:
#             json.dump(sample,Dev)
#             Dev.write(', ')
#         elif count <= 8407:
#             json.dump(sample,Tst)
#             Tst.write(', ')
#
#         count+=1
from keras.layers.core import Reshape, Permute, Dense, Flatten, Lambda
from keras.layers.merge import Concatenate, Multiply, dot, Dot
import numpy as np
scene = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[2,2,2]]])
scene = Permute((2, 1))(Concatenate()(scene))