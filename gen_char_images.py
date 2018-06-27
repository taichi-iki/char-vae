# coding utf-8

"""
Written_by: Taichi Iki
Created_at: 2018-06-25
Comment:
This script generates images related to the characters contained in the specified font file.
"""

import pickle
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw


class ArgSpace(object):
    def __init__(args, parse_from_sys):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--font', type=str,
                help='path & name of font',
                default='GenEiAntique-4.2/GenEiAntique_v4.ttc'
            )
        parser.add_argument('--save', type=str,
                help='path & name of output file',
                default='char_images.pklb'
            )
        ns = parser.parse_args(args=None if parse_from_sys else '')
        
        args.font_pathname = ns.font
        args.save_pathname = ns.save
        args.img_size = 64
                    

def append_special_characters(target_dict, img_size):
    target_dict['<UNK>'] = np.ones(shape=(img_size, img_size)).astype('float32')
    target_dict['<EOS>'] = np.ones(shape=(img_size, img_size)).astype('float32')
    target_dict['<EOS>'][:, :5] = 0.0
    target_dict['<BOS>'] = np.ones(shape=(img_size, img_size)).astype('float32')
    target_dict['<BOS>'][:,-5:] = 0.0


def get_glyphs_as_charlist(font_pathname):
    font_number = 0
    charlist = []
    while True:
        ttf = TTFont(font_pathname, fontNumber=font_number)
        glyphs = ttf.getGlyphSet().keys()

        for table in ttf["cmap"].tables:
            filtered = filter(lambda c: c[1] in glyphs, table.cmap.items())
            charlist.extend([chr(c[0]) for c in filtered])
        
        ttf.close()
        
        font_number += 1
        if font_number >= ttf.reader.numFonts:
            break

    charlist = sorted(set(charlist))
    return charlist


def main(args):
    print('Making a character list from %s ...'%args.font_pathname)
    charlist = get_glyphs_as_charlist(args.font_pathname)
    
    print('fetched %d characters'%len(charlist))
    print('Outputing font images...')
    font_size = int(args.img_size*0.85)
    font = ImageFont.truetype(args.font_pathname, font_size)

    char_image_dict = {}
    append_special_characters(char_image_dict, args.img_size)
    
    for c in charlist:
        img = Image.new('RGB', size=(args.img_size, args.img_size))
        
        if c == '\t' or c == '\n':
            # preventing space-like characters from being drawn
            pass
        else:
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), c, font=font)

        char_image_dict[c] = (np.asarray(img).astype('float32')[:,:,0])/255.0
        img.close()
    
    print('Generating a character image dictionary as %s ...'%args.save_pathname)
    with open(args.save_pathname, 'wb') as f:
        pickle.Pickler(f).dump(char_image_dict)
    
    print('done')
    

if __name__ == '__main__':
    args = ArgSpace(parse_from_sys=True)
    main(args)
