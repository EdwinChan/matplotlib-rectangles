This module allows the layout of `matplotlib` plots to be specified in a
precise way. Layouts can be nested, so complicated layouts can be constructed
from simpler ones. The module is capable of creating any layout that can be
described by a system of linear equations, but helper methods are provided to
simplify the task of building commonly used layouts.

The file `mplrect.py` can be imported as a module. The file can also be run
directly to show an example of a layout. The example provides a starting point
for creating similar layouts. For simple use cases, it suffices to modify the
predefined lengths in `PredefinedLayout`, `ScreenLayout`, `PrintLayout`, and
`PresentationLayout`.

The module requires Python 3.
