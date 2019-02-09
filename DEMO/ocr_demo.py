import pytesseract as ocr
from PIL import Image
import cv2
import numpy as np

def read(imagem):
    '''convertendo em um array editável de numpy[x, y, CANALS]'''
    npimagem = np.asarray(imagem.convert('RGB')).astype(np.uint8)  

    '''diminuição dos ruidos antes da binarização'''
    npimagem[:, :, 0] = 0 # zerando o canal R (RED)
    npimagem[:, :, 2] = 0 # zerando o canal B (BLUE)

    '''atribuição em escala de cinza'''
    im = cv2.cvtColor(npimagem, cv2.COLOR_RGB2GRAY) 

    '''
    aplicação da truncagem binária para a intensidade
    pixels de intensidade de cor abaixo de 127 serão convertidos para 0 (PRETO)
    pixels de intensidade de cor acima de 127 serão convertidos para 255 (BRANCO)
    A atrubição do THRESH_OTSU incrementa uma análise inteligente dos nivels de truncagem
    '''    
    ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 

    '''reconvertendo o retorno do threshold em um objeto do tipo PIL.Image'''
    binimagem = Image.fromarray(thresh) 

    '''chamada ao tesseract OCR por meio de seu wrapper'''
    phrase = ocr.image_to_string(binimagem, lang='por')
    return 


for i in range(5):
    imagem = Image.open(f'doc{i}.png')
    phrase = ocr.image_to_string(imagem, lang='por')
    print(phrase, end='\n\n################\n\n')