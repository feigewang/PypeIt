import os
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.transforms as mtransforms
from matplotlib.widgets import Button, Slider
from scipy.interpolate import RectBivariateSpline

matplotlib.use('Qt5Agg')

from pypeit import specobjs
from pypeit.par import pypeitpar
from pypeit.core.wavecal import fitting, waveio, wvutils
from pypeit import msgs

operations = dict({'cursor': "Select lines (LMB click)\n" +
                    "         Select regions (LMB drag = add, RMB drag = remove)\n" +
                    "         Navigate (LMB drag = pan, RMB drag = zoom)",
                   'p' : "Toggle pan/zoom with the cursor",
                   '?' : "Display the available options",
                   'q' : "Close Object ID window and continue PypeIt reduction",
                   })


class ObjFindGUI(object):
    """
    GUI to interactively identify object traces. The GUI can be run within
    PypeIt during data reduction, or as a standalone script outside of
    PypeIt. To initialise the GUI, call the initialise() function in this
    file.
    """

    def __init__(self, canvas, image, axes, sobjs):
        """Controls for the interactive Object ID tasks in PypeIt.

        The main goal of this routine is to interactively add/delete
        object apertures.

        Args:
            canvas (Matploltib figure canvas): The canvas on which all axes are contained
            image (ndarray): The image plotted to screen
            axes (dict): Dictionary of four Matplotlib axes instances (Main spectrum panel, two for residuals, one for information)
            sobjs (SpecObjs): An instance of the SpecObjs class
        """
        # Store the axes
        self.image = image
        self.axes = axes
        self.specobjs = sobjs
        self.objtraces = []
        self._obj_idx = -1

        # Unset some of the matplotlib keymaps
        matplotlib.pyplot.rcParams['keymap.fullscreen'] = ''        # toggling fullscreen (Default: f, ctrl+f)
        matplotlib.pyplot.rcParams['keymap.home'] = ''              # home or reset mnemonic (Default: h, r, home)
        matplotlib.pyplot.rcParams['keymap.back'] = ''              # forward / backward keys to enable (Default: left, c, backspace)
        matplotlib.pyplot.rcParams['keymap.forward'] = ''           # left handed quick navigation (Default: right, v)
        #matplotlib.pyplot.rcParams['keymap.pan'] = ''              # pan mnemonic (Default: p)
        matplotlib.pyplot.rcParams['keymap.zoom'] = ''              # zoom mnemonic (Default: o)
        matplotlib.pyplot.rcParams['keymap.save'] = ''              # saving current figure (Default: s)
        matplotlib.pyplot.rcParams['keymap.quit'] = ''              # close the current figure (Default: ctrl+w, cmd+w)
        matplotlib.pyplot.rcParams['keymap.grid'] = ''              # switching on/off a grid in current axes (Default: g)
        matplotlib.pyplot.rcParams['keymap.yscale'] = ''            # toggle scaling of y-axes ('log'/'linear') (Default: l)
        matplotlib.pyplot.rcParams['keymap.xscale'] = ''            # toggle scaling of x-axes ('log'/'linear') (Default: L, k)
        matplotlib.pyplot.rcParams['keymap.all_axes'] = ''          # enable all axes (Default: a)

        # Initialise the main canvas tools
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas = canvas

        # Interaction variables
        self._respreq = [False, None]  # Does the user need to provide a response before any other operation will be permitted? Once the user responds, the second element of this array provides the action to be performed.
        self._qconf = False  # Confirm quit message
        self._changes = False

        # Draw the spectrum
        self.canvas.draw()

    def print_help(self):
        """Print the keys and descriptions that can be used for Identification
        """
        keys = operations.keys()
        print("===============================================================")
        print("       OBJECT ID OPERATIONS")
        for key in keys:
            print("{0:6s} : {1:s}".format(key, operations[key]))
        print("---------------------------------------------------------------")

    def replot(self):
        """Redraw the entire canvas
        """
        self.canvas.restore_region(self.background)
        self.draw_objtraces()
        self.canvas.draw()

    def draw_objtraces(self):
        """Draw the lines and annotate with their IDs
        """
        for i in self.objtraces: i.pop(0).remove()
        self.objtraces = []
        # Plot the object traces
        for iobj in range(self.specobjs.nobj):
            if iobj == 0:
                spectrc = np.arange(self.specobjs[0].trace_spat.size)
            if iobj == self._obj_idx:
                self.objtraces.append(self.axes['main'].plot(self.specobjs[iobj].trace_spat, spectrc,
                                                             'r-', linewidth=2, alpha=0.5))
            else:
                self.objtraces.append(self.axes['main'].plot(self.specobjs[iobj].trace_spat, spectrc,
                                                             'r--', linewidth=1, alpha=0.5))

    def draw_callback(self, event):
        """Draw the lines and annotate with their IDs

        Args:
            event (Event): A matplotlib event instance
        """
        # Get the background
        self.background = self.canvas.copy_from_bbox(self.axes['main'].bbox)
        self.draw_objtraces()

    def get_ind_under_point(self, event):
        """Get the index of the line closest to the cursor

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            ind (int): Index of the spectrum where the event occurred
        """
        ind = np.argmin(np.abs(self.specx - event.xdata))
        return ind

    def get_axisID(self, event):
        """Get the ID of the axis where an event has occurred

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            axisID (int, None): Axis where the event has occurred
        """
        if event.inaxes == self.axes['main']:
            return 0
        elif event.inaxes == self.axes['info']:
            return 1
        return None

    def mouse_move_callback(self, event):
        """
        Get the index of the spectrum closest to the cursor
        """
        if event.inaxes is None:
            return
        axisID = self.get_axisID(event)
        if axisID is not None:
            if axisID == 0 and event.button == 3:
                self.mmx, self.mmy = event.xdata, event.ydata
                self.mouseidx = self.get_ind_under_point(event)

    def button_press_callback(self, event):
        """What to do when the mouse button is pressed

        Args:
            event (Event): Matplotlib event instance containing information about the event
        """
        if event.inaxes is None:
            return
        if self.canvas.toolbar.mode != "":
            return
        if event.button == 1:
            self._addsub = 1
        elif event.button == 3:
            self._addsub = 0
        axisID = self.get_axisID(event)
        self._start = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        """What to do when the mouse button is released

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            None
        """
        if event.inaxes is None:
            return
        if event.inaxes == self.axes['info']:
            if (event.xdata > 0.8) and (event.xdata < 0.9):
                answer = "y"
            elif event.xdata >= 0.9:
                answer = "n"
            else:
                return
            self.operations(answer, -1)
            self.update_infobox(default=True)
            return
        elif self._respreq[0]:
            # The user is trying to do something before they have responded to a question
            return
        if self.canvas.toolbar.mode != "":
            return
        # Draw an actor
        axisID = self.get_axisID(event)
        if axisID is not None:
            if axisID <= 2:
                self._end = self.get_ind_under_point(event)
                if self._end == self._start:
                    # The mouse button was pressed (not dragged)
                    pass
                elif self._end != self._start:
                    # The mouse button was dragged
                    if axisID == 0:
                        if self._start > self._end:
                            tmp = self._start
                            self._start = self._end
                            self._end = tmp
                        # Now do something
                        pass
        # Now plot
        self.canvas.restore_region(self.background)
        self.draw_objtraces()
        self.canvas.draw()

#self.image.set_clim(vmin=10, vmax=100)

    def key_press_callback(self, event):
        """What to do when a key is pressed

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            None
        """
        # Check that the event is in an axis...
        if not event.inaxes:
            return
        # ... but not the information box!
        if event.inaxes == self.axes['info']:
            return
        axisID = self.get_axisID(event)
        self.operations(event.key, axisID)

    def operations(self, key, axisID):
        """Canvas operations

        Args:
            key (str): Which key has been pressed
            axisID (int): The index of the axis where the key has been pressed (see get_axisID)
        """
        # Check if the user really wants to quit
        if key == 'q' and self._qconf:
            if self._changes:
                self.update_infobox(message="WARNING: There are unsaved changes!!\nPress q again to exit", yesno=False)
                self._qconf = True
            else:
                msgs.bug("Need to change this to kill and return the results to PypeIt")
                plt.close()
        elif self._qconf:
            self.update_infobox(default=True)
            self._qconf = False

        # Manage responses from questions posed to the user.
        if self._respreq[0]:
            if key != "y" and key != "n":
                return
            else:
                # Switch off the required response
                self._respreq[0] = False
                # Deal with the response
                if self._respreq[1] == "write":
                    # First remove the old file, and save the new one
                    msgs.work("Not implemented yet!")
                else:
                    return
            # Reset the info box
            self.update_infobox(default=True)
            return

        if key == '?':
            self.print_help()
        elif key == 'q':
            if self._changes:
                self.update_infobox(message="WARNING: There are unsaved changes!!\nPress q again to exit", yesno=False)
                self._qconf = True
            else:
                plt.close()
        self.canvas.draw()

    def add_object(self):
        thisobj = specobjs.SpecObj(frameshape, slit_spat_pos, slit_spec_pos,
                                   det=specobj_dict['det'],
                                   setup=specobj_dict['setup'], slitid=specobj_dict['slitid'],
                                   orderindx=specobj_dict['orderindx'],
                                   objtype=specobj_dict['objtype'])
        thisobj.hand_extract_spec = hand_extract_spec[iobj]
        thisobj.hand_extract_spat = hand_extract_spat[iobj]
        thisobj.hand_extract_det = hand_extract_det[iobj]
        thisobj.hand_extract_fwhm = hand_extract_fwhm[iobj]
        thisobj.hand_extract_flag = True
        f_ximg = RectBivariateSpline(spec_vec, spat_vec, ximg)
        thisobj.spat_fracpos = f_ximg(thisobj.hand_extract_spec, thisobj.hand_extract_spat,
                                      grid=False)  # interpolate from ximg
        thisobj.smash_peakflux = np.interp(thisobj.spat_fracpos * nsamp, np.arange(nsamp),
                                           fluxconv_cont)  # interpolate from fluxconv
        # assign the trace
        spat_0 = np.interp(thisobj.hand_extract_spec, spec_vec, trace_model)
        shift = thisobj.hand_extract_spat - spat_0
        thisobj.trace_spat = trace_model + shift
        thisobj.trace_spec = spec_vec
        thisobj.spat_pixpos = thisobj.trace_spat[specmid]
        thisobj.set_idx()
        if hand_extract_fwhm[iobj] is not None:  # If a hand_extract_fwhm was input use that for the fwhm
            thisobj.fwhm = hand_extract_fwhm[iobj]
        elif nobj_reg > 0:  # Otherwise is None was input, then use the median of objects on this slit if they are present
            thisobj.fwhm = med_fwhm_reg
        else:  # Otherwise just use the fwhm parameter input to the code (or the default value)
            thisobj.fwhm = fwhm
        # Finish
        sobjs.add_sobj(thisobj)

    def update_infobox(self, message="Press '?' to list the available options",
                       yesno=True, default=False):
        """Send a new message to the information window at the top of the canvas

        Args:
            message (str): Message to be displayed
        """
        self.axes['info'].clear()
        if default:
            self.axes['info'].text(0.5, 0.5, "Press '?' to list the available options", transform=self.axes['info'].transAxes,
                          horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
        # Display the message
        self.axes['info'].text(0.5, 0.5, message, transform=self.axes['info'].transAxes,
                      horizontalalignment='center', verticalalignment='center')
        if yesno:
            self.axes['info'].fill_between([0.8, 0.9], 0, 1, facecolor='green', alpha=0.5, transform=self.axes['info'].transAxes)
            self.axes['info'].fill_between([0.9, 1.0], 0, 1, facecolor='red', alpha=0.5, transform=self.axes['info'].transAxes)
            self.axes['info'].text(0.85, 0.5, "YES", transform=self.axes['info'].transAxes,
                          horizontalalignment='center', verticalalignment='center')
            self.axes['info'].text(0.95, 0.5, "NO", transform=self.axes['info'].transAxes,
                          horizontalalignment='center', verticalalignment='center')
        self.axes['info'].set_xlim((0, 1))
        self.axes['info'].set_ylim((0, 1))
        self.canvas.draw()

def initialise(frame, trace_dict, sobjs, slit_ids=None):
    """Initialise the 'ObjFindGUI' window for interactive object tracing

        Args:
            frame (ndarray): Sky subtracted science image
            trace_dict (dict): Dictionary containing slit and object trace information
            sobjs (SpecObjs): SpecObjs Class
            slit_ids (list, None): List of slit ID numbers

        Returns:
            ofgui (ObjFindGUI): Returns an instance of the ObjFindGUI class
    """
    # This allows the input lord and rord to either be (nspec, nslit) arrays or a single
    # vectors of size (nspec)
    if trace_dict['edges_l'].ndim == 2:
        nslit = trace_dict['edges_l'].shape[1]
        lordloc = trace_dict['edges_l']
        rordloc = trace_dict['edges_r']
    else:
        nslit = 1
        lordloc = trace_dict['edges_l'].reshape((trace_dict['edges_l'].size, 1))
        rordloc = trace_dict['edges_r'].reshape((trace_dict['edges_r'].size, 1))

    # Assign the slit IDs if none were provided
    if slit_ids is None:
        slit_ids = [str(slit) for slit in np.arange(nslit)]

    # Determine the scale of the image
    med = np.median(frame)
    mad = np.median(np.abs(frame-med))
    vmin = med-3*mad
    vmax = med+3*mad

    # Add the main figure axis
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    plt.subplots_adjust(bottom=0.05, top=0.85, left=0.05, right=0.85)
    image = ax.imshow(frame, aspect=frame.shape[1]/frame.shape[0], cmap = 'Greys', vmin=vmin, vmax=vmax)

    # Overplot the slit traces
    specarr = np.arange(lordloc.shape[0])
    for sl in range(nslit):
        ax.plot(lordloc[:, sl], specarr, 'g-')
        ax.plot(rordloc[:, sl], specarr, 'b-')

    # Add an information GUI axis
    axinfo = fig.add_axes([0.15, .92, .7, 0.07])
    axinfo.get_xaxis().set_visible(False)
    axinfo.get_yaxis().set_visible(False)
    axinfo.text(0.5, 0.5, "Press '?' to list the available options", transform=axinfo.transAxes,
                horizontalalignment='center', verticalalignment='center')
    axinfo.set_xlim((0, 1))
    axinfo.set_ylim((0, 1))

    axes = dict(main=ax, info=axinfo)
    # Initialise the object finding window and display to screen
    fig.canvas.set_window_title('PypeIt - Object Tracing')
    ofgui = ObjFindGUI(fig.canvas, image, axes, sobjs)
    plt.show()

    return ofgui


def temp_load(fname):
    import pickle
    with open(fname, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    dirname = "/Users/rcooke/Work/Research/vmp_DLAs/observing/WHT_ISIS_2019B/N1/wht_isis_blue_U/"
    frame = np.load(dirname+"frame.npy")
    sobjs = temp_load(dirname+"sobjs.pkl")
    trace_dict = temp_load(dirname+"trace_dict.pkl")
    initialise(frame, trace_dict, sobjs, slit_ids=None)