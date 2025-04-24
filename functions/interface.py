

import tkinter as TK
from tkinter import PhotoImage, Tk, Canvas
from tkinter.filedialog import askopenfilename
import PIL


model = ''
NbPps = 0
def window1(model, NbPps):
    window = TK.Tk()
    window.title("Parameters Choice")
    
    TK.Label(window, text = 'Number of participants:').grid(row = 1, column = 0, sticky = 'e', padx=5, pady=5)
    TK.Label(window, text = 'Model').grid(row = 2, column = 0, sticky = 'e', padx=5, pady=5)
    
    VarNbPps = TK.StringVar()
    VarModel = TK.StringVar()
    
    
    TK.Entry(window, textvariable = VarNbPps).grid(row = 1, column = 1, sticky = 'w')
    TK.Entry(window, textvariable = VarModel).grid(row = 2, column = 1, sticky = 'w')
    
    def printing():
        global NbPps, model
        if VarNbPps.get() != '' and VarModel.get() != '':
            NbPps = VarNbPps.get()
            model = VarModel.get()
            print("Nb pps : {} & Modele : {}".format(NbPps, model))
    
    TK.Button(window, text='OK', command = printing).grid(row = 3, column = 1, rowspan=2, sticky = 'w e n s')
    window.mainloop()
    return (model, NbPps)
    


# features_list = TK.Listbox(window)
# final_features_list = TK.Listbox(window)

# TK.Button(window, text = "Charger 📥", command = charger).grid(row=1, column=0, columnspan=1, sticky = 'w', padx = 5, pady = 5)


# def charger():
#     final_features_list.delete(0, 'end')
#     features = all_data.columns.to_list()
#     for ft in features:
#         features_list.insert('end', ft)
 
# def features_choice():
#     final_features_list.delete(0, 'end')
    
#     featureslist = TK.Tk()   
#     featureslist.title("Features")
    
#     List = TK.Listbox(featureslist)
    
#     TK.Label(features_list, text='List of features').grid(row=0, column = 0, sticky = 'w', padx = 5, pady = 5)
#     features_list.grid.grid(row=0, column = 0, sticky = 'w', padx = 5, pady = 5)
#     creation_features_list = TK.Listbox(features_list)
    
#     def selecting_one_feature(event):
       
#         position_selectionne = features_list.curselection()[0] #retourne un t-uple
#         feature_selectionnee = features[position_selectionne]
#         if len(L)<=len(features):
#             creation_features_list.insert('end', feature_selectionnee)
#             final_features_list.insert('end', feature_selectionne)
#             LL.insert('end', feature_selectionne)
#             # Liste_objets_featureslists[0]._get_membres().append(feature_selectionne)
         
#         else :
#             erreur = TK.Tk()
#             erreur.title('Erreur')
#             TK.Label(erreur, text = "Already the max feature selected.").grid(row = 0, column = 0, sticky = 'w e n s', padx = 5, pady = 5 )
        
#     List.bind('<Double-Button-1>', selecting_one_feature)
    
#     def deleting_one_feature(event):
        
#         position_selectionne = creation_features_list.curselection()[0] #retourne un t-uple
#         creation_features_list.delete(creation_features_list.curselection()[0])
#         final_features_list.delete(creation_features_list.curselection()[0])
#         del L[position_selectionne]
#         del Liste_objets_featureslists[0]._get_memebres()[position_selectionne]
#         LL.delete(creation_features_list.curselection()[0])
        
#     TK.Label(featureslist, text='Creating the list of features').grid(row=0, column = 2, sticky = 'e', padx = 5, pady = 5)
#     creation_features_list.grid(row = 1, column = 2, sticky = 'e', padx = 5, pady = 5)
    
   
#     creation_features_list.bind('<Double-Button-1>', deleting_one_feature)
    
#     TK.Button(featureslist, text="Load 📥", command = jouer).grid(row = 2, column = 3, sticky = 'w e n s', padx = 5, pady = 5)
 