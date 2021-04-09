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
from scipy.stats import mode
import seaborn as sns
import pandas as pd
import itertools
import functools
from toolz.itertoolz import sliding_window


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
        self.waitmax = 10
        # max distance between para in consecutive frames in xy or xz vector space.
        self.thresh = 5
        self.double = False
#        self.area = area
        self.avg_velocity = 0

    def endrecord(self, clip):
        # CHOPS OFF THE HANGING LEN(WAITMAX) PORTION WHERE IT COULNDT FIND PARTNER
        if clip:
            self.location = self.location[:-self.waitmax]
        self.avg_velocity = np.mean(
            [np.linalg.norm(p) for p in np.diff(self.location, axis=0)])
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
                self.endrecord(True)
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
            self.endrecord(True)
            self.double = True
        return contlist


class ParaMaster2D():
    def __init__(self, directory, startframe, endframe):
        self.framerate = 100
        self.directory = directory
        self.all_xy_para = []
        self.xy_matrix = []
        self.distance_thresh = 100
        self.length_thresh = 50
        self.paravectors = []
        self.dots = []
        self.analyzed_frames = deque()
        self.analyzed_frames_raw = deque()
        self.af_mod = 5
        self.velocity_mags = []
        self.interp_indices = []
        self.sec_per_br_frame = 1
        self.frames_per_mode = 20
        self.backgrounds = []
        self.framewindow = [startframe, endframe]
        self.pvid_id = [file_id for file_id in
                        os.listdir(self.directory) if file_id[-4:] == '.AVI'][0]
        self.total_numframes = 0
        self.US_frames = (np.loadtxt(directory + "/" +
                                     [file_id for file_id in
                                      os.listdir(self.directory) if file_id[-12:] ==
                                      'US_times.txt'][0]) / 10).astype(np.int)
        self.CS_frames = (np.loadtxt(directory + "/" + 
                                     [file_id for file_id in
                                      os.listdir(self.directory) if file_id[-12:] ==
                                      'CS_times.txt'][0]) / 10).astype(np.int)
        
    def exporter(self):
        print('exporting ParaMaster')
        with open('paradata2D.pkl', 'wb') as file:
            pickle.dump(self, file)

# This function fits contours to each frame of the high contrast videos created in flparse2. Contours are filtered for proper paramecia size, then the locations of contours are compared to the current locations of paramecia. Locations of contours falling within a distance threshold of a known paramecia are appended to the location list for that para. At the end, individual paramecia records are established for each paramecia in the tank. Each para object is stored in the all_xy or all_xz para object list depending on plane.

    def contrast_frame(self, frame_id, grayframe, prms):
    # obtains correct background window
        br = self.backgrounds[
            int(np.floor(frame_id / (self.framerate *
                                     self.sec_per_br_frame *
                                     self.frames_per_mode)))].astype(np.uint8)
        frame_avg = np.mean(grayframe)
        br_avg = np.mean(br)
        img_adj = (grayframe * (br_avg / frame_avg)).astype(np.uint8)
#        brsub = cv2.absdiff(img_adj, br)
        brsub = cv2.absdiff(grayframe.astype(np.uint8), br)
        cont_ = imcont(brsub, prms).astype(np.uint8)
        return cont_

    def make_backgrounds(self):
        print("IN BACKGROUNDS")
        curr_frame = cv2.CAP_PROP_POS_FRAMES
        fc = cv2.CAP_PROP_FRAME_COUNT
        pvid_vcap = cv2.VideoCapture(self.directory + "/" + self.pvid_id)
        self.total_numframes = pvid_vcap.get(fc)
        ir_br_frames_to_mode = range(1,
                                     int(self.total_numframes),
                                     int(self.sec_per_br_frame * self.framerate))
        ir_temp = []
        for frame in ir_br_frames_to_mode:
            pvid_vcap.set(curr_frame, frame)
            ret, im = pvid_vcap.read()
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ir_temp.append(img)
            if len(ir_temp) % self.frames_per_mode == 0:
                self.backgrounds.append(
                    calc_mode(ir_temp,
                              np.zeros([1024, 1280])))
                ir_temp = []
        # accounts for leftovers at the end
        if ir_temp:
            self.backgrounds.append(calc_mode(ir_temp,
                                                np.zeros([1024, 1280])))
            ir_temp = []
        np.save(self.directory + '/backgrounds.npy', self.backgrounds)
        pvid_vcap.release()

    def findpara(self, params):
        print("FINDING PARA")
        # filtering_params give erosion, dilation, threshold, and area for para finding
        self.analyzed_frames = deque()
        completed_para_records = []
        filtering_params, area = params
        curr_frame = cv2.CAP_PROP_POS_FRAMES
        fc = cv2.CAP_PROP_FRAME_COUNT
        pvid_vcap = cv2.VideoCapture(self.directory + "/" + self.pvid_id)
        self.total_numframes = pvid_vcap.get(fc)
        firstframe = True
        pvid_vcap.set(curr_frame, self.framewindow[0])
        for framenum in range(self.framewindow[0], self.framewindow[1]):
            if framenum % 100 == 0:
                print(framenum)
            ret, im = pvid_vcap.read()
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_contrasted = self.contrast_frame(framenum, im_gray,
                                                filtering_params)
            im_cont_color = cv2.cvtColor(im_contrasted, cv2.COLOR_GRAY2RGB)
            rim, contours, hierarchy = cv2.findContours(
                im_contrasted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parafilter = [cv2.minEnclosingCircle(t)[0] for t in contours
                          if area[0] <= cv2.contourArea(t) < area[1]]
            for para in parafilter:
                cv2.circle(im_cont_color, (int(para[0]), int(para[1])), 2,
                           (255, 0, 255), -1)
            if framenum % self.af_mod == 0:
                self.analyzed_frames.append(im_cont_color)
            if firstframe:
                firstframe = False
                para_objects = [Para(framenum, pr) for pr in parafilter]
            else:
                xy_list = list(map(lambda n: n.nearby(parafilter), para_objects))
                newpara = [Para(framenum, pr) for pr in parafilter]
                para_objects = para_objects + newpara
                paramecia = list(filter(lambda x: x.completed, para_objects))
                completed_para_records = completed_para_records + paramecia
                para_objects = list(filter(lambda x: not x.completed, para_objects))
#current para list p_t and p_s are cleansed of records that are complete.
        self.watch_event()
        self.analyzed_frames_raw = copy.deepcopy(self.analyzed_frames)
        pvid_vcap.release()
        modify = input('Modify Videos?: ')
        if modify == 'r':
            modify = self.repeat_vids()
        if modify == 's':
            self.startover = True
            self.framewindow = [self.framewindow[0], int(self.total_numframes)]
            return self.findpara(params)
        if modify == 'y':
            action = input('Enter new params: ')
            self.analyzed_frames = deque()
            return self.findpara(ast.literal_eval(action))
        if modify == 'x':
            return ""
        else:
            end_precs = list(map(lambda p: p.endrecord(False), para_objects))
            all_xy_para = completed_para_records + para_objects
            all_xy_para = sorted(all_xy_para, key=lambda x: len(x.location))
            all_xy_para.reverse()
            self.all_xy_para_raw = all_xy_para
            self.all_xy_para = [para for para in all_xy_para
                                if len(para.location) > self.length_thresh]
            print('Para Found in XY')
            print(len(self.all_xy_para))
            self.create_coord_matrix()
            self.label_para()

    def repeat_vids(self):
        self.watch_event()
        ra = input('Repeat, Yes, or No: ')
        if ra == 'r':
            return self.repeat_vids()
        elif ra == 'y':
            return 'y'
        else:
            return 'n'

    def refilter_para_records(self, reclen, avg_vel, xb, yb, xtravel):
        all_xy_records = copy.deepcopy(self.all_xy_para_raw)
        # this has to come first b/c p.location has to be nonzero
        all_xy_records = list(filter(lambda p: len(p.location) > reclen, all_xy_records))
        all_xy_records = list(filter(lambda p: p.avg_velocity > avg_vel, all_xy_records))
        all_xy_records = list(filter(
             lambda p: xb[0] < np.mean(np.array(p.location)[:, 0]) < xb[1], all_xy_records))
        all_xy_records = list(filter(
             lambda p: yb[0] < np.mean(np.array(p.location)[:, 1]) < yb[1], all_xy_records))
        # want an x cumulative travel cutoff.
        all_xy_records = list(filter(lambda p: np.ptp(np.array(p.location)[:,0]) > xtravel, all_xy_records))
        self.all_xy_para = all_xy_records
        self.recalculate_mat_and_labels()
    
    def watch_event(self, *make_movie):
        self.analyzed_frames = copy.deepcopy(self.analyzed_frames_raw)
        self.label_para()
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        pvid_vw = cv2.VideoWriter('pvid_id_movie.AVI', fourcc, 30,
                                  (1280, 1024), True)
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        for fr, im in enumerate(self.analyzed_frames):
            cv2.imshow('vid', im)
            key_entered = cv2.waitKey(1)
            if key_entered == 50:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)


    def show_labeled_ir_frames(self):
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        curr_frame = cv2.CAP_PROP_POS_FRAMES
        fc = cv2.CAP_PROP_FRAME_COUNT
        pvid_vcap = cv2.VideoCapture(self.directory + "/" + self.pvid_id)
        for frame in range(self.framewindow[0], self.framewindow[1]):
            ret, im = pvid_vcap.read()
            if frame % 10 == 0:
                for para_id, xyr in enumerate(self.xy_matrix[:, frame]):
                    # indicates a nan.
                    if math.isnan(xyr[0]):
                        pass
                    else:
                        cv2.putText(
                            im,
                            str(para_id),
                            (int(xyr[0]+7),
                             int(xyr[1])),
                            0,
                            .5,
                            color=(255, 0, 255))
                    if any(map(lambda x: x < frame < x + 200, self.US_frames)):
                        cv2.putText(
                            im,
                            "US ON",
                            (20, 40),
                            0,
                            1.5,
                            color=(0, 255, 255),
                            thickness=3)
                    if any(map(lambda x: x < frame < x + 200, self.CS_frames)):
                        cv2.putText(
                            im,
                            "CS ON",
                            (20, 40),
                            0,
                            1.5,
                            color=(0, 125, 125),
                            thickness=3)
                cv2.imshow('vid', im)
                key_entered = cv2.waitKey(1)
                if key_entered == 50:
                    break
                elif key_entered == 49:
                    cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        
    def label_para(self):
        temp_frames = deque()
        frame_num = 0
        while True:
            try:
                im = self.analyzed_frames.popleft()
            except IndexError:
                break
            if frame_num % 100 == 0:
                print(frame_num)
            para_id = 0
            # the row index of the para3Dcooords matrix defines xyz id.
            for para_id, xyr in enumerate(self.xy_matrix[:, frame_num]):
                # indicates a nan.
                if math.isnan(xyr[0]):
                    pass
                else:
                    cv2.putText(
                        im,
                        str(para_id),
                        (int(xyr[0]+5),
                         int(xyr[1])),
                        0,
                        .5,
                        color=(255, 255, 0))
            temp_frames.append(im)
            frame_num += self.af_mod
        self.analyzed_frames = temp_frames

    def recalculate_mat_and_labels(self):
        self.create_coord_matrix()

    def merge_by_ycoord(self, ycrange):
        in_y_range = list(map(
            lambda p: ycrange[0] < np.mean(np.array(p.location)[:,1]) < ycrange[1],
            self.all_xy_para))
        print(in_y_range)
        if sum(in_y_range) <= 1:
            self.recalculate_mat_and_labels()
            return 1
        else:
            p1 = in_y_range.index(True)
            p2 = in_y_range[p1+1:].index(True) + p1
            self.merge_records(p1, p2, False)
            return self.merge_by_ycoord(ycrange)

    def merge_records(self, rec1, rec2, relabel):
        p1 = self.all_xy_para[rec1]
        p2 = self.all_xy_para[rec2]
        # this is correct
        time_win1 = p1.timestamp + len(p1.location)
        if time_win1-1 == p2.timestamp:
            p2loc = p2.location[1:]
        else:
            p2loc = p2.location
        p1.location = p1.location + [np.array([np.nan, np.nan]) for i in range(
            time_win1, p2.timestamp)] + p2loc
        del self.all_xy_para[rec2]
        if relabel:
            self.recalculate_mat_and_labels()
            
    def manual_remove(self, reclist, relabel):
        reclist.sort()
        reclist.reverse()
        for rec in reclist:
            del self.all_xy_para[rec]
        if relabel:
            self.recalculate_mat_and_labels()

    def create_coord_matrix(self):
        xy_matrix = np.zeros([len(self.all_xy_para),
                              int(self.total_numframes)],
                             dtype='2f')
        for row, rec in enumerate(self.all_xy_para):
            # this is correct
            xy_coords = [(np.nan, np.nan) for i in range(int(self.total_numframes))]
            xy_coords[rec.timestamp:rec.timestamp+len(
                rec.location)] = rec.location
            xy_matrix[row] = xy_coords
        self.xy_matrix = xy_matrix
        np.save(self.directory + '/para_matrix.npy', xy_matrix)


def plot_x_tsplot(pmaster):
    # need random color from para object and the xy_matrix
    # dataframe needs to be like ParaID, frame, xcoord, ycoord
    fig, ax = pl.subplots(1, 2)
    df_dict = {'frames': list(itertools.chain.from_iterable([
        range(pmaster.xy_matrix.shape[1]) for i in range(pmaster.xy_matrix.shape[0])])),
               'xcoords': list(itertools.chain.from_iterable([x for x in pmaster.xy_matrix[:,:, 0]])),
               'ycoords': list(itertools.chain.from_iterable([x for x in pmaster.xy_matrix[:,:, 1]])),
               'para_id': list(itertools.chain.from_iterable([i*np.ones(pmaster.xy_matrix.shape[1]) for i in range(pmaster.xy_matrix.shape[0])]))}
    df = pd.DataFrame(df_dict)
    fig, ax = pl.subplots()
    sns.lineplot(data=df, x='frames', y='xcoords', ax=ax)
    for usframe, csframe in zip(pmaster.US_frames, pmaster.CS_frames):
        ax.vlines(x=usframe, ymin=0, ymax=1024, color='yellow')
        ax.vlines(x=csframe, ymin=0, ymax=1024, color='gray')
    pl.show()
    return df


def plot_all_recs(pmaster):
    fig, ax = pl.subplots()
    for xy_rec in pmaster.xy_matrix:
        xcoords = xy_rec[:,0]
        avg_y_coord = np.nanmean(xy_rec[:,1])
        ax.plot([x+avg_y_coord for x in xcoords])
    for usframe, csframe in zip(pmaster.US_frames, pmaster.CS_frames):
        ax.vlines(x=usframe, ymin=0, ymax=2000, color='yellow')
        ax.vlines(x=csframe, ymin=0, ymax=2000, color='gray')
    pl.show()

    
def us_cs_responses(pmaster):
    fig, ax = pl.subplots(2, len(pmaster.CS_frames))
    baseline = 100
    bounds = 100
    for c_ind, csf in enumerate(pmaster.CS_frames):
        x_from_cs = []
        for xyr in pmaster.xy_matrix:
            x_from_cs.append(np.cumsum(np.diff(xyr[csf-baseline:csf+200, 0])))
        ts_plot(x_from_cs, ax[0, c_ind], 'frame', 'x')
        ax[0, c_ind].vlines(x=baseline, ymin=-bounds, ymax=bounds, color='gray')
    for u_ind, usf in enumerate(pmaster.US_frames):
        x_from_us = []
        for xyr in pmaster.xy_matrix:
            x_from_us.append(np.cumsum(np.diff(xyr[usf-baseline:usf+200, 0])))
        ts_plot(x_from_us, ax[1, u_ind], 'frame', 'x')
        ax[1, u_ind].vlines(x=baseline, ymin=-bounds, ymax=bounds, color='yellow')
    pl.show()
            


def ts_plot(list_of_lists, ax, lab_x, lab_y):
    index_list = list(
        itertools.chain.from_iterable(
            [range(len(arr)) for arr in list_of_lists]))
    id_list = [(ind*np.ones(len(arr))).tolist() for ind, arr in enumerate(list_of_lists)]
    ids_concatenated = list(itertools.chain.from_iterable(id_list))
    #this works if passed np.arrays instead of lists
    value_list = list(itertools.chain.from_iterable(list_of_lists))
    df_dict = {lab_x: index_list, 
               lab_y: value_list}
    df = pd.DataFrame(df_dict)
    sns.lineplot(data=df, x=lab_x, y=lab_y, ax=ax)
    return df


def imcont(image, params):
    thresh, erode_win, dilate_win, med_win = params
    r, th = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    ek = np.ones([erode_win, erode_win]).astype(np.uint8)
    dk = np.ones([dilate_win, dilate_win]).astype(np.uint8)
    er = cv2.erode(th, ek)
    dl = cv2.dilate(er, dk)
    md = cv2.medianBlur(dl, med_win)
    return md


def calc_median(deq, nump_arr):
    # so k are the values, j are the indicies.
    for j, k in enumerate(nump_arr[:, 0]):
        nump_arr[j] = np.median([x[j] for x in deq], axis=0)
    return nump_arr


def calc_mode(deq, nump_arr):
    for j, k in enumerate(nump_arr[:, 0]):
        nump_arr[j, :] = mode(np.array([x[j, :] for x in deq]))[0]
    return nump_arr


def automerge_records(pmaster):
    timethresh = 1000
    spacethresh = 100
    for ind, para in enumerate(pmaster.all_xy_para):
        base_ts = para.timestamp+len(para.location)-1
        base_pos = para.location[-1]
        near_and_prox = list(map(lambda cp: (0 <= cp.timestamp-base_ts < timethresh) and (np.linalg.norm(cp.location[0] - base_pos) < spacethresh), pmaster.all_xy_para))
        try:
            cand_match = near_and_prox.index(True)
            if cand_match == ind:
                cand_match = near_and_prox[ind+1:](True)
        except ValueError:
            continue
        pmaster.merge_records(ind, cand_match, False)
        return automerge_records(pmaster)
    pmaster.recalculate_mat_and_labels()


def make_paramaster(directory,
                    start_ind,
                    end_ind):
    paramaster = ParaMaster2D(directory, start_ind, end_ind)
    try:
        paramaster.backgrounds = np.load(directory+"/backgrounds.npy")
    except:
        paramaster.make_backgrounds()
    paramaster.findpara([[12, 1, 3, 3], [10, 500]])
    return paramaster


def find_paravectors():
    return 1


if __name__ == '__main__':
    dir_input = input("Enter Directory:  ")
    directory = "E:/ParaBehaviorData/" + dir_input
    pmaster = make_paramaster(directory,
                              1,
                              500)
    pmaster.refilter_para_records(300, .25, [140, 1050], [40, 950], 50)
    automerge_records(pmaster)

#    [12, 1, 3, 3]

