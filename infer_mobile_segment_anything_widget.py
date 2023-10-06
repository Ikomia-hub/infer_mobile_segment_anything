# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_mobile_segment_anything.infer_mobile_segment_anything_process import InferMobileSegmentAnythingParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferMobileSegmentAnythingWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferMobileSegmentAnythingParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        self.check_cuda = pyqtutils.append_check(self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        self.spin_points_per_side = pyqtutils.append_spin(self.grid_layout,
                                                          "Points per side",
                                                          self.parameters.points_per_side,
                                                          min=1)

        self.spin_input_size_percent = pyqtutils.append_spin(self.grid_layout,
                                                          "Image size (%)",
                                                          self.parameters.input_size_percent,
                                                          min=1, max=100)
 
        self.spin_mask_output = pyqtutils.append_spin(self.grid_layout,
                                                          "Mask ID (If graphic input set)",
                                                          self.parameters.mask_id,
                                                          min=1, max=3)
        self.edit_box_input = pyqtutils.append_edit(self.grid_layout,
                                                    "Box coord. xyxy (optional)",
                                                    self.parameters.input_box
                                                    )
        
        self.edit_point_input = pyqtutils.append_edit(self.grid_layout,
                                            "Point coord. xy (optional)",
                                            self.parameters.input_point
                                            )
        
        self.edit_point_label = pyqtutils.append_edit(self.grid_layout,
                                    "Point label (optional)",
                                    self.parameters.input_point_label
                                    )

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.points_per_side = self.spin_points_per_side.value()
        self.parameters.input_size_percent = self.spin_input_size_percent.value()
        self.parameters.mask_id = self.spin_mask_output.value()
        self.parameters.input_box = self.edit_box_input.text()
        self.parameters.input_point = self.edit_point_input.text()
        self.parameters.input_point_label = self.edit_point_label.text()
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferMobileSegmentAnythingWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_mobile_segment_anything"

    def create(self, param):
        # Create widget object
        return InferMobileSegmentAnythingWidget(param, None)
