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

    def endrecord(self):
        # CHOPS OFF THE HANGING LEN(WAITMAX) PORTION WHERE IT COULNDT FIND PARTNER
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
#            parafilter = [cv2.minEnclosingCircle(t)[0] for t in contours
 #                         if area[0] <= cv2.contourArea(t) < area[1]]
            for para in parafilter:
                # cv2.circle(im_cont_color, (int(para[0][0]), int(para[0][1])), 2,
                #            (255, 0, 255), -1)
                cv2.circle(im_cont_color, (int(para[0]), int(para[1])), 2,
                           (255, 0, 255), -1)
            self.analyzed_frames.append(im_cont_color)
#            self.analyzed_frames.append(im)
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
        # Once you get this right, give option to extend analysis until the end of the movie. 
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

    #This function is the most important for paramecia matching in both planes. First, an empty correlation matrix is created with len(all_xy) columns and len(all_xz) rows. The datatype is a 3-element tuple that takes a pearson coefficient, a pvalue, and a time of overlap for each para object in all_xy vs all_xz. I initialize the corrmat with 0,1 and [] for each element.

    def refilter_para_records(self, area, reclen, avg_vel):
        all_xy_records = self.all_xy_para_raw
        all_xy_records = list(filter(lambda p: len(p.location) > reclen, all_xy_records))
        all_xy_records = list(filter(lambda p: avg_vel < p.avg_velocity, all_xy_records))
        self.all_xy_para = all_xy_records
        self.create_coord_matrix()
        self.analyzed_frames = copy.deepcopy(self.analyzed_frames_raw)
        self.label_para()

    
    def watch_event(self, *make_movie):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        pvid_vw = cv2.VideoWriter('pvid_id_movie.AVI', fourcc, 30,
                                  (1280, 1024), True)

        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        for fr, im in enumerate(self.analyzed_frames):
#            im = cv2.resize(im, (800, 800))
            if fr % 5 == 0:
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
                            color=(255, 255, 0))
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
            frame_num += 1
        self.analyzed_frames = temp_frames

    def manual_merge(self, rec1, rec2):
        p1 = self.all_xy_para[rec1]
        p2 = self.all_xy_para[rec2]
        time_win1 = p1.timestamp + len(p1.location)
        p1.location = p1.location + [np.nan for i in range(
            time_win1, p2.timestamp)] + p2.location
        del self.all_xy_para[rec2]
        self.analyzed_frames = copy.deepcopy(self.analyzed_frames_raw)
        self.label_para()

    def manual_remove(self, reclist):
        reclist.sort()
        reclist.reverse()
        for rec in reclist:
            del self.all_xy_para[rec]
        self.analyzed_frames = copy.deepcopy(self.analyzed_frames_raw)
        self.label_para()

    def create_coord_matrix(self):
        xy_matrix = np.zeros([len(self.all_xy_para),
                              int(self.total_numframes)],
                             dtype='2f')
        for row, rec in enumerate(self.all_xy_para):
            xy_coords = [(np.nan, np.nan) for i in range(int(self.total_numframes))]
#            xy_location_w_inv_y = [(x, 1024-y) for (x, y) in rec.location]
            # xy_coords[rec.timestamp:rec.timestamp+len(
            #     rec.location)] = xy_location_w_inv_y
            xy_coords[rec.timestamp:rec.timestamp+len(
                rec.location)] = rec.location
            xy_matrix[row] = xy_coords
        self.xy_matrix = xy_matrix


def plot_xy_trajectories(pmaster):
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


def ts_plot(list_of_lists):
    fig, ax = pl.subplots(1, 1)
    index_list = list(
        itertools.chain.from_iterable(
            [range(len(arr)) for arr in list_of_lists]))
    id_list = [(ind*np.ones(len(arr))).tolist() for ind, arr in enumerate(list_of_lists)]
    ids_concatenated = list(itertools.chain.from_iterable(id_list))
    #this works if passed np.arrays instead of lists
    value_list = list(itertools.chain.from_iterable(list_of_lists))
    df_dict = {'x': index_list, 
               'y': value_list}
    df = pd.DataFrame(df_dict)
    sns.lineplot(data=df, x='x', y='y', ax=ax)
    pl.show()
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


def make_paramaster(directory,
                    start_ind,
                    end_ind):
    paramaster = ParaMaster2D(directory, start_ind, end_ind)
#    paramaster.make_backgrounds()
    paramaster.backgrounds = np.load(directory+"/backgrounds.npy")
    paramaster.findpara([[3, 3, 5, 3], [10, 500]])
    return paramaster


def find_paravectors():
    return 1


def make_2D_animation():
    states = np.concatenate([[x, x, x] for x in states])
    sm_states = smooth_states(5, states)
    print("Length of Para Record")
    print(len(para_x))
    fig = pl.figure(figsize=(12, 8))
    p_xy_ax = fig.add_subplot(131)
    p_xy_ax.set_title('XY COORDS')
    p_xy_ax.set_xlim([0, 1888])
    p_xy_ax.set_ylim([0, 1888])
    p_xz_ax = fig.add_subplot(132)
    p_xz_ax.set_title('XZ COORDS')
    p_xz_ax.set_xlim([0, 1888])
    p_xz_ax.set_ylim([0, 1888])
    p_xy_ax.set_aspect('equal')
    p_xz_ax.set_aspect('equal')
    state_ax = fig.add_subplot(133)
    state_ax.set_ylim([-.5, 3.5])
    state_ax.set_xlim([0, len(para_x)])
    state_ax.set_title('PARA STATE')
    state_ax.set_aspect('auto', 'box-forced')
    def updater(num, plotlist):
        if num < 1:
            return plotlist
        px = para_x[num]
        py = para_y[num]
        pz = para_z[num]
        px_hist = para_x[0:num]
        py_hist = para_y[0:num]
        pz_hist = para_z[0:num]
        state_x = range(num)
        state_y = states[0:num]
#        smoothed_state_y = sm_states[0:num]
        plotlist[2].set_data(px_hist, py_hist)
        plotlist[3].set_data(px_hist, pz_hist)
        plotlist[0].set_data(px, py)
        plotlist[1].set_data(px, pz)
        plotlist[4].set_data(state_x, state_y)
#        plotlist[3].set_data(state_x, smoothed_state_y)
        return plotlist
    p_xy_line, = p_xy_ax.plot([], [], linewidth=1, color='g')
    p_xz_line, = p_xz_ax.plot([], [], linewidth=1, color='g')
    p_xy_plot, = p_xy_ax.plot([], [], marker='.', ms=15, color='m')
    p_xz_plot, = p_xz_ax.plot([], [], marker='.', ms=15, color='m')
    state_plot, = state_ax.plot([], [], marker='.', linestyle='None', color='k')
  #  sm_state_plot, = state_ax.plot([], [], linewidth=1.0)
    p_list = [p_xy_plot, p_xz_plot, p_xy_line, p_xz_line, state_plot]
  # sm_state_plot]
    line_ani = anim.FuncAnimation(
        fig,
        updater,
        len(para_x),
        fargs=[p_list],
        interval=2,
        repeat=True,
        blit=True)
#        line_ani.save('test.mp4')
    pl.show()

if __name__ == '__main__':

    a = 1
    # GOOD STARTING PARAMS
    pmaster = make_paramaster("/Users/nightcrawler2/Dropbox/rikers1",
                              1,
                             500)
    

