from src.utils import *
from vtkmodules.vtkRenderingCore import vtkPointPicker


def create_crosshair_actors(crosshair_length):
    """
    Creates crosshairs across all actors for selected slices
    :param crosshair_length: (float) value for length of crosshair

    :return Pyvista actors
    """
    actors = {}
    for axis in ['x', 'y', 'z']:
        line_source = vtk.vtkLineSource()
        if axis == 'x':
            line_source.SetPoint1([-crosshair_length / 2, 0, 0])
            line_source.SetPoint2([crosshair_length / 2, 0, 0])
        elif axis == 'y':
            line_source.SetPoint1([0, -crosshair_length / 2, 0])
            line_source.SetPoint2([0, crosshair_length / 2, 0])
        elif axis == 'z':
            line_source.SetPoint1([0, 0, -crosshair_length / 2])
            line_source.SetPoint2([0, 0, crosshair_length / 2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 0)  # Yellow color
        actors[axis] = (line_source, actor)  # Store both line_source and actor

    return actors


def update_crosshairs(position, plotter, crosshair_actors, crosshair_length):
    """
    Update the position of the crosshairs for a keyboard event
    :param position: 3D position for interactive display
    :param plotter: Pyvista plotter
    :param crosshair_actors: Pyvista actors for crosshairs 
    :param crosshair_length: (float) length for crosshairs
    """
    for axis, (line_source, actor) in crosshair_actors.items():
        if axis == 'x':
            line_source.SetPoint1([position[0] - crosshair_length / 2, position[1], position[2]])
            line_source.SetPoint2([position[0] + crosshair_length / 2, position[1], position[2]])
        elif axis == 'y':
            line_source.SetPoint1([position[0], position[1] - crosshair_length / 2, position[2]])
            line_source.SetPoint2([position[0], position[1] + crosshair_length / 2, position[2]])
        elif axis == 'z':
            line_source.SetPoint1([position[0], position[1], position[2] - crosshair_length / 2])
            line_source.SetPoint2([position[0], position[1], position[2] + crosshair_length / 2])

    plotter.update()


def set_opacity(plotter, slice_actors_list, target_slice_actor, clicked_slice_idx, actor_no_opacity=None):
    """
    Function to set the correct opacity, to highlight only relevant slices in all windows

    :param plotter: Pyvista plotter
    :param slice_actors_list: (list) of Pyvista actors
    :param target_slice_actor: Pyvista actor with data at every slice (acquired 3D volume)
    :param actor_no_opacity: Pyvista actor where only the plotted slice is to be visible
    :param clicked_slice_idx: index of the slice to highlight
    """
    for slice_idx, actor in target_slice_actor.items():
        opacity = 1.0 if slice_idx == clicked_slice_idx else 0.1
        actor.GetProperty().SetOpacity(opacity)
        for slice_actor in slice_actors_list:
            if slice_idx in slice_actor:
                if isinstance(slice_actor[slice_idx], list):
                    for actor in slice_actor[slice_idx]:
                        actor.GetProperty().SetOpacity(opacity)
                else:
                    slice_actor[slice_idx].GetProperty().SetOpacity(opacity)

        if actor_no_opacity is not None:
            if slice_idx in actor_no_opacity:
                if slice_idx != clicked_slice_idx:
                    actor_no_opacity[slice_idx].GetProperty().SetOpacity(0)
                else:
                    actor_no_opacity[slice_idx].GetProperty().SetOpacity(opacity)

    plotter.update()


def click_callback(widget, event_name, plotter, volume, target_slice_actor, target_opacity_actor,
                   slice_actors_list, crosshair_actors, crosshair_length, current_slice_idx,
                   actor_no_opacity=None):
    """
    Finds the closest slices to the interaction (click event)
    :param plotter: (Pyvsita plotter)
    :param volume: (SimpleITK volume) containing spacing characteristics of the data
    :param target_slice_actor: (Pyvista actor) used as target for determining slice index
    :param target_opacity_actor: (Pyvista actor) used as target for setting opacity
    :param slice_actors_list: (list) of Pyvista actors to be plotted
    :param crosshair_actors: Pyvista actors for crosshairs
    :param crosshair_length: (float) length of crosshairs
    :param current_slice_idx: (int) starting slice index
    :param actor_no_opacity: (Pyvista actor) where only the highlighted slice should be displayed
    """

    closest_slice_idx, min_distance = None, float('inf')

    x, y = plotter.iren.interactor.GetEventPosition()

    # Set up the point picker and click callback
    picker = vtkPointPicker()
    picker.Pick(x, y, 0, plotter.renderer)
    picked_position = picker.GetPickPosition()

    for slice_idx, actors in target_slice_actor.items():
        slice_center = actors.GetMapper().GetCenter()

        distance = np.linalg.norm(np.array(slice_center) - np.array(picked_position))
        if distance < min_distance:
            min_distance = distance
            closest_slice_idx = slice_idx

    if closest_slice_idx is not None:
        current_slice_idx[0] = closest_slice_idx  # Update current slice index
        set_opacity(plotter, slice_actors_list, target_opacity_actor, closest_slice_idx, actor_no_opacity)
        update_crosshairs([0, 0, closest_slice_idx * volume.GetSpacing()[0]],
                          plotter, crosshair_actors, crosshair_length)


def keyboard_callback(widget, event_name, plotter, volume, target_slice_actor,
                      slice_actors_list, crosshair_actors, crosshair_length, current_slice_idx,
                      actor_no_opacity=None):
    """
    Finds the closest slices to the interaction (keyboard event)
    :param plotter: (Pyvsita plotter)
    :param volume: (SimpleITK volume) containing spacing characteristics of the data
    :param target_slice_actor: (Pyvista actor) used as target for determining slice index and setting opacity
    :param slice_actors_list: (list) of Pyvista actors to be plotted
    :param crosshair_actors: Pyvista actors for crosshairs
    :param crosshair_length: (float) length of crosshairs
    :param current_slice_idx: (int) starting point for slice index
    :param actor_no_opacity: (Pyvista actor) where only the highlighted slice should be displayed
    """
    key = plotter.iren.interactor.GetKeySym()

    if key == "j":
        current_slice_idx[0] = min(current_slice_idx[0] + 1, max(target_slice_actor.keys()))
        set_opacity(plotter, slice_actors_list, target_slice_actor, current_slice_idx[0], actor_no_opacity)
        update_crosshairs([0, 0, current_slice_idx[0] * volume.GetSpacing()[0]],
                          plotter, crosshair_actors, crosshair_length)

    elif key == "k":
        current_slice_idx[0] = max(current_slice_idx[0] - 1, min(target_slice_actor.keys()))
        set_opacity(plotter, slice_actors_list, target_slice_actor, current_slice_idx[0], actor_no_opacity)
        update_crosshairs([0, 0, current_slice_idx[0] * volume.GetSpacing()[0]], plotter,
                          crosshair_actors, crosshair_length)
