import ast
import os
import cv2
import imageio
from matplotlib import pyplot as pl
import numpy as np
import pickle
import math
import scipy.ndimage
import toolz
from scipy.stats.stats import pearsonr
import copy
import matplotlib.animation as anim
from collections import deque
import seaborn
import matplotlib.gridspec as gridspec



class Para:
    def __init__(self, timestmp, coord):
        self.location = [coord]
        self.lastcoord = coord
        self.timestamp = timestmp
        self.color = [np.random.random(), np.random.random(), np.random.random(
        )]
        self.completed = False
        self.waitindex = 0
        # waits at max 4 frames before giving up on particular para.
        self.waitmax = 4
        # max distance between para in consecutive frames in xy or xz vector space.
        self.thresh = 20
        self.double = False

    def endrecord(self):
        # CHOPS OFF THE HANGING LEN(WAITMAX) PORTION WHERE IT COULNDT FIND PARTNER
        self.location = self.location[:-self.waitmax]
        self.completed = True

    def nearby(self, contlist):
        #first, search the list of contours found in the image to see if any of them are near the previous position of this Para.
        if len(contlist) == 0:
            pcoords = np.array([])
        else:
            cont_arr = np.array(contlist)
            lastc = np.reshape(self.lastcoord, (1, 2))
            distance_past_thresh = np.where(
                np.sqrt(
                    np.sum(
                        (cont_arr-lastc)*(cont_arr-lastc), axis=1)) < self.thresh)
            pcoords = cont_arr[distance_past_thresh]
        
#if there's nothing found, add 1 to the waitindex, and say current position is the last position

        if pcoords.shape[0] == 0:
            self.location.append(self.lastcoord)
            if self.waitindex == self.waitmax:
                #end record if you've gone 'waitmax' frames without finding anything. this value greatly changes things. its a delicate balance between losing the para and waiting too long while another para enters
                self.endrecord()
            self.waitindex += 1

# this case is only one contour is within the threshold distance to this Para.
        elif pcoords.shape[0] == 1:
            newcoord = pcoords[0]
            index_to_pop = distance_past_thresh[0][0]
            self.location.append(newcoord)
            self.lastcoord = newcoord
            self.waitindex = 0
            contlist.pop(index_to_pop)

# this case is that two or more contours fit threshold distance. stop the record and mark it as a double.
        elif pcoords.shape[0] > 1:       
            self.endrecord()
            self.double = True
        return contlist


class ParaMaster2D():
    def __init__(self, directory):
        self.framerate = 100
        self.directory = directory
        self.decimated_vids = False
        self.startover = False
        self.all_xy_para = []
        self.distance_thresh = 100
        self.length_thresh = 30
        self.time_thresh = 60
        self.filter_width = 5
        self.paravectors = []
        self.paravectors_normalized = []
        self.dots = []
        self.makemovies = False
        self.analyzed_frames = deque()
        self.analyzed_frames_original = deque()
        self.dotprod = []
        self.velocity_mags = []
        self.length_map = np.vectorize(lambda a: a.shape[0])
        self.interp_indices = []
        self.sec_per_br_frame = 2
        self.frames_per_mode = 20
        self.backgrounds = []
        self.framewindow = [1, -1]
        self.pvid_id = [file_id for file_id in
                        os.listdir(directory) if file_id[-4:] == '.AVI'][0]
        
    def exporter(self):
        print('exporting ParaMaster')
        with open('paradata2D.pkl', 'wb') as file:
            pickle.dump(self, file)

# This function fits contours to each frame of the high contrast videos created in flparse2. Contours are filtered for proper paramecia size, then the locations of contours are compared to the current locations of paramecia. Locations of contours falling within a distance threshold of a known paramecia are appended to the location list for that para. At the end, individual paramecia records are established for each paramecia in the tank. Each para object is stored in the all_xy or all_xz para object list depending on plane.

    def contrast_frame(self, frame_id, grayframe, prms):
    # obtains correct background window
        br = self.backgrounds[np.floor(frame_id / (self.framerate *
                                                   self.sec_per_br_frame *
                                                   self.frames_per_mode))]
        brsub = brsub_img(grayframe, br)
        cont_ = imcont(brsub, prms).astype(np.uint8)
        return cont_

    def make_backgrounds(self):
        curr_frame = cv2.CAP_PROP_POS_FRAMES
        fc = cv2.CAP_PROP_FRAME_COUNT
        pvid_vcap = cv2.VideoCapture(data_directory + pvid_id)
        numframes = pvid_vcap.get(fc)
        ir_br_frames_to_mode = range(self.framewindow[0], numframes,
                                     self.sec_per_br_frame * self.framerate)
        ir_temp = []
        ir_backgrounds = []
        for frame in ir_br_frames_to_mode:
            pvid_vcap.set(curr_frame, i)
            ret, im = pvid_vcap.read()
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ir_temp.append(img)
            if len(ir_temp) % self.frames_per_mode == 0:
                self.backgrounds.append(
                    calc_median(ir_temp,
                                np.zeros([1280, 1024])))
                ir_temp = []
        # accounts for leftovers at the end
        if ir_temp:
            self.backgrounds.append(calc_median(ir_temp,
                                                np.zeros([1280, 1024])))
            ir_temp = []
        np.save(data_directory + 'backgrounds.npy', self.backgrounds)

    def findpara(self, params):
        # filtering_params give erosion, dilation, threshold, and area for para finding
        filtering_params, area = params
        paramecia = []
        completed_para_records = []
        pvid_vcap = cv2.VideoCapture(self.directory + self.pvid_id)

        for framenum in range(1, self.framewindow[1]):
            ret, im = pvid_vcap.read()
            im_contrasted = contrast_frame(im,
                                           framenum,
                                           filtering_params)
            im_contrasted_color = cv2.cvtColor(im_contrasted, cv2.COLOR_GRAY2RGB)
            self.analyzed_frames.append(im_contrasted)
            rim, contours, hierarchy = cv2.findContours(
                cont_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parafilter = [cv2.minEnclosingCircle(t)[0] for t in contours
                          if area[0] <= cv2.contourArea(t) < area[1]]
            for para in parafilter:
                cv2.circle(im_contrasted_color, (int(para[0]), int(para[1])), 3,
                           (0, 0, 255), -1)
            if firstframe:
                firstframe = False
                para_objects = [Para(framenum, pr) for pr in parafilter]
# p_t is a list of para objects. asks if any elements of contour list are nearby each para p.
            else:
                xylist = map(lambda n: n.nearby(parafilter), para_objects)
                newpara = [Para(framenum, cord) for cord in parafilter]
                para_objects = para_objects + newpara
                complete_records = filter(lambda x: x.completed, para_objects)
                completed_para_records = completed_para_recors + complete_records
                para_objects = filter(lambda x: not x.completed, para_objects)
#current para list p_t and p_s are cleansed of records that are complete.
        self.watch_event()

        modify = raw_input('Modify Videos?: ')
        if modify == 'r':
            modify = repeat_vids()
        if modify == 'q':
            self.startover = True
            return
        if modify == 'y':
            action = raw_input('Enter new params: ')
            self.analyzed_frames = deque()
            pvid_vcap.release()
            return self.findpara(ast.literal_eval(action))
        else:
            all_xy_para = completed_para_records + para_objects
            all_xy_para = sorted(all_xy_para, key=lambda x: len(x.location))
            all_xy_para.reverse()
            self.all_xy_para = [para for para in all_xy_para
                                if len(para.location) > 0]
            self.long_xy = [para for para in self.all_xy_para
                            if len(para.location) >= self.length_thresh]
            print('Para Found in XY')
            print(len(self.all_xy_para))
            self.analyzed_frames_original = copy.deepcopy(self.analyzed_frames)
            self.label_para()
            pvid_vcap.release()




    #This function is the most important for paramecia matching in both planes. First, an empty correlation matrix is created with len(all_xy) columns and len(all_xz) rows. The datatype is a 3-element tuple that takes a pearson coefficient, a pvalue, and a time of overlap for each para object in all_xy vs all_xz. I initialize the corrmat with 0,1 and [] for each element.

    def clear_frames(self):
        self.analyzed_frames = copy.deepcopy(self.analyzed_original)
        
    
    def manual_match(self):
        xy = 0
        xz = 0
        key = raw_input("Fix?: ")
        if key == 'y':
            xy = raw_input('Enter XY rec #  ')
            xz = raw_input('Enter XZ rec #  ')
            xy = int(xy)
            xz = int(xz)
            pl.close()
            for ind_xy, up_xy in enumerate(self.unpaired_xy):
                if up_xy[0] == xy:
                    del self.unpaired_xy[ind_xy]
                    break
            for ind_xz, up_xz in enumerate(self.unpaired_xz):
                if up_xz[0] == xz:
                    del self.unpaired_xz[ind_xz]
                    break
            self.xyzrecords.append([[xy, xz, (1, 1, 500)]])
            self.clear_frames()
            return 1
        else:
            pl.close()
            return 0

    def make_id_movie(self):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        pvid_vw = cv2.VideoWriter('pvid_id_movie.AVI', fourcc, 30,
                                  (1280, 1024), True)
        while True:
            try:
                im = self.analyzed_frames.popleft()
                pvid_vw.write(im)
            except IndexError:
                break
        pvid_vw.release()

        
    def watch_event(self):
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        for im in self.analyzed_frames:
            im = cv2.resize(im, (700, 700))
            cv2.imshow('vid', im)
            key_entered = cv2.waitKey(15)
            if key_entered == 50:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def label_para(self):
        temp_frames = deque()
        frame_num = 1
        while True:
            try:
                im = self.analyzed_frames.popleft()
            except IndexError:
                break
            if frame_num % 100 == 0:
                print(frame_num)
            para_id = 0
            # the row index of the para3Dcooords matrix defines xyz id.
            for xyr in self.all_xy_para:
                # indicates a nan.
                if math.isnan(xyr[2][frame_num][0]):
                    pass
                else:
                    cv2.putText(
                        im,
                        str(xyr[0]),
                        (int(xyr[2][frame_num][0]),
                         int(1024 - xyr[2][frame_num][1])),
                        0,
                        1,
                        color=(255, 0, 255))
                para_id += 1
            temp_frames.append(im)
            frame_num += 1
        self.analyzed_frames = temp_top


    def manual_merge(self, rec1, rec2):
        p1 = self.all_xy_para[rec1]
        p2 = self.all_xy_para[rec2]
        time_win1 = p1.timestamp + len(p1.location)
        p1.location = p1.location + [np.nan for i in range(time_win1, p2.timestamp)] + p2.location
        del self.all_xy_para[rec2]
        self.clear_frames()
        self.label_para()
            
    
def imcont(image, params):
    thresh, erode_win, dilate_win, med_win = params
    r, th = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    ek = np.ones([erode_win, erode_win]).astype(np.uint8)
    dk = np.ones([dilate_win, dilate_win]).astype(np.uint8)
    er = cv2.erode(th, ek)
    dl = cv2.dilate(er, dk)
    md = cv2.medianBlur(dl, med_win)
    return md

# mean norm gets rid of luminance diffs between frames by looking at tank edge
# illumination
def brsub_img(img, ir_br):
    brsub = cv2.absdiff(img, ir_br)
    return brsub

def return_paramaster_object(start_ind,
                             end_ind,
                             makemovies, directory, showstats, pcw):
    paramaster = ParaMaster(start_ind, end_ind, directory, pcw)
    if makemovies:
        paramaster.makemovies = True
    paramaster.parawrapper(showstats)
    return paramaster

def calc_median(deq, nump_arr):
    # so k are the values, j are the indicies.
    for j, k in enumerate(nump_arr[:, 0]):
        nump_arr[j] = np.median([x[j] for x in deq], axis=0)
    return nump_arr




if __name__ == '__main__':

    # GOOD STARTING PARAMS
    self.findpara([[10, 3, 5, 3], [10, 3, 5, 3], 6], False, True)
    
    pmaster = return_paramaster_object(1000,
                                       1599,
                                       False,
                                       os.getcwd() + '/Fish00/',
                                       False,
                                       600)
#     paramaster = ParaMaster(1499, 1550, os.getcwd() + '/Fish00/')
#     paramaster.makemovies = False
#     paramaster.parawrapperq(False)
# #    paramaster.find_paravectors(False)
