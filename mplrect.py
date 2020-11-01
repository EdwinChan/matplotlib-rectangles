import collections.abc
import itertools
import numbers
import numpy
import numpy.linalg
import uuid

def check_grid_size(rects, nx, ny):
  if nx*ny != len(rects):
    raise ValueError(
      'grid of size ({}, {}) cannot be filled with {} rectangles'
      .format(nx, ny, len(rects)))

def delayed(iterable):
  yield
  yield from iter(iterable)

def split_every(iterable, count):
  return itertools.zip_longest(*[iter(iterable)] * count)

def all_same(iterable):
  iterable = iter(iterable)
  head = next(iterable, object())
  return all(head == item for item in iterable)

def check_arg(name, value, choices):
  if value not in choices:
    raise ValueError('{} must be in {}'.format(name, repr(choices)))

class Dimension:
  def __init__(self, value, unit):
    self.value = value
    self.unit = unit

  def __pos__(self):
    return Dimension(self.value, self.unit)

  def __neg__(self):
    return Dimension(-self.value, self.unit)

  def __add__(self, other):
    if isinstance(other, Dimension):
      return Dimension(self.inch_value() + other.inch_value(), 'in')
    if isinstance(other, str):
      return self + Dimension.parse(other)
    return NotImplemented

  def __sub__(self, other):
    if isinstance(other, Dimension):
      return Dimension(self.inch_value() - other.inch_value(), 'in')
    if isinstance(other, str):
      return self - Dimension.parse(other)
    return NotImplemented

  def __mul__(self, other):
    if isinstance(other, numbers.Real):
      return Dimension(self.value * other, self.unit)
    return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, numbers.Real):
      return Dimension(self.value / other, self.unit)
    return NotImplemented

  def __rmul__(self, other):
    return self * other

  def parse(string):
    import re
    match = re.match(r'^\s*(.*?)\s*([a-z]+)\s*$', string)
    if match:
      return Dimension(float(match.group(1)), match.group(2))
    raise ValueError('dimension {!r} is not understood'.format(string))

  def inch_value(self):
    unit_conversions = {'in': 1, 'cm': 1/2.54, 'pt': 1/72}
    for unit, conversion in unit_conversions.items():
      if self.unit == unit:
        return self.value * conversion
    raise ValueError('unit {!r} is not understood'.format(string))

Dimension.zero = Dimension(0, 'in')

class Length:
  inch = uuid.uuid4()

  def __init__(self, repr):
    self.repr = repr

  def __pos__(self):
    return Length(self.repr.copy())

  def __neg__(self):
    return Length({var: -coeff for var, coeff in self.repr.items()})

  def __add__(self, other):
    if isinstance(other, Length):
      repr = self.repr.copy()
      for var, coeff in other.repr.items():
        repr[var] = repr.get(var, 0) + coeff
      return Length(repr)
    if isinstance(other, Dimension):
      return self + Length({Length.inch: other.inch_value()})
    if isinstance(other, str):
      return self + Dimension.parse(other)
    return NotImplemented

  def __sub__(self, other):
    if isinstance(other, Length):
      repr = self.repr.copy()
      for var, coeff in other.repr.items():
        repr[var] = repr.get(var, 0) - coeff
      return Length(repr)
    if isinstance(other, Dimension):
      return self - Length({Length.inch: other.inch_value()})
    if isinstance(other, str):
      return self - Dimension.parse(other)
    return NotImplemented

  def __mul__(self, other):
    if isinstance(other, numbers.Real):
      return Length({var: coeff * other for var, coeff in self.repr.items()})
    return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, numbers.Real):
      return Length({var: coeff / other for var, coeff in self.repr.items()})
    return NotImplemented

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return -self + other

  def __rmul__(self, other):
    return self * other

class Rectangle:
  """
  A rectangle possibly containing other rectangles.

  The coordinates of the top-left and bottom-right corners of the
  rectangle are (self.x1, self.y1) and (self.x2, self.y2)
  respectively.
  """

  def __init__(self):
    self.rects = []
    self.vars = [uuid.uuid4() for _ in range(4)]
    self.eqns = []
    self.x1 = Length({self.vars[0]: 1})
    self.y1 = Length({self.vars[1]: 1})
    self.x2 = Length({self.vars[2]: 1})
    self.y2 = Length({self.vars[3]: 1})
    self.dx = self.x2 - self.x1
    self.dy = self.y2 - self.y1

  def add(self, rect):
    """
    Add a child rectangle to this rectangle.

    Adding a child rectangle has the effect of notifying this
    rectangle that it is responsible for determining the position
    and size of the child rectangle by solving the constraints added
    to this rectangle. Only lengths constructed from lengths of
    children rectangles can appear in a constraint.

    A child rectangle may contain grandchildren rectangles; this
    allows layouts to be built recursively.
    """
    if rect in self.rects:
      raise ValueError('rectangle has already been added')
    self.rects.append(rect)

  def set_equal(self, *lengths):
    """
    Add a constraint that two lengths be equal.

    Only lengths constructed from lengths of children rectangles can
    appear in a constraint.
    """
    if not lengths:
      raise ValueError('at least two lengths must be given')
    head, *tail = lengths
    if isinstance(head, collections.abc.Iterable):
      if tail:
        raise ValueError('more than one iterable argument is given')
      head, *tail = head
    for length in tail:
      self.eqns.append(head - length)

  def size_to_grid(self, rects, nx, ny, sx=Dimension.zero, sy=Dimension.zero):
    """
    Arrange rectangles on a grid.

    The grid has nx rows and ny columns. The function ensures that
    each rectangle is separated from its horizontal neighbors by sx
    and from its vertical neighbors by sy. The function further
    ensures that rectangles sharing a row have the same dy, and
    rectangles sharing a column have the same dx. No constraints are
    placed on the sizes of individual rectangles. The caller should
    impose constraints on dx of any one row of rectangles, and dy on
    any one column of rectangles.

    If a single value of sx is given, it is used for the spacing
    between all columns. If a list is given, the first value is the
    spacing between the first and second columns, the second value
    is the spacing between the second and third columns, and so on;
    the list is cycled as necessary. The situation is analogous for
    sy.
    """
    check_grid_size(rects, nx, ny)
    sx, sy = self.ensure_list(sx), self.ensure_list(sy)

    for i, sx_ in zip(range(nx), delayed(itertools.cycle(sx))):
      for j, sy_ in zip(range(ny), delayed(itertools.cycle(sy))):
        rect = rects[nx*j+i]
        if j > 0:
          above_rect = rects[nx*(j-1)+i]
          self.set_equal(rect.dx, above_rect.dx)
        if i > 0:
          left_rect = rects[nx*j+(i-1)]
          self.set_equal(rect.dy, left_rect.dy)
        if j > 0:
          self.set_equal(rect.x1, above_rect.x1)
          self.set_equal(rect.y1, above_rect.y2 + sy_)
        elif i > 0:
          self.set_equal(rect.x1, left_rect.x2 + sx_)
          self.set_equal(rect.y1, left_rect.y1)

  def anchor_to_grid(self, rects, nx, ny, sx, sy):
    """
    Arrange the top-left corners of some rectangles on a grid.

    The grid has nx rows and ny columns. The function ensures that
    the top-left corner of each rectangle is separated from the same
    corner of its horizontal neighbors by sx, and from the same
    corner of its vertical neighbors by sy. No constraints are
    placed on the sizes of the rectangles.

    If a single value of sx is given, it is used for the spacing
    between all columns. If a list is given, the first value is the
    spacing between the first and second columns, the second value
    is the spacing between the second and third columns, and so on;
    the list is cycled as necessary. The situation is analogous for
    sy.
    """
    check_grid_size(rects, nx, ny)
    sx, sy = self.ensure_list(sx), self.ensure_list(sy)

    for i, sx_ in zip(range(nx), delayed(itertools.cycle(sx))):
      for j, sy_ in zip(range(ny), delayed(itertools.cycle(sy))):
        rect = rects[nx*j+i]
        if j > 0:
          above_rect = rects[nx*(j-1)+i]
          self.set_equal(rect.x1, above_rect.x1)
          self.set_equal(rect.y1, above_rect.y1 + sy_)
        elif i > 0:
          left_rect = rects[nx*j+(i-1)]
          self.set_equal(rect.x1, left_rect.x1 + sx_)
          self.set_equal(rect.y1, left_rect.y1)

  def copy(self):
    """
    Copy this rectangle.

    All children rectangles are copied as well. The copied
    rectangles are distinct from the original ones.
    """
    copy = Rectangle()
    for rect in self.rects:
      copy.add(rect.copy())
    var_map = dict(zip(self.iter_vars(), copy.iter_vars()))
    var_map[Length.inch] = Length.inch
    copy.eqns = [
      Length({var_map[var]: coeff for var, coeff in eqn.repr.items()})
      for eqn in self.eqns]
    return copy

  def solve(self):
    """
    Determine the layout that satisfies the constraints.

    Constraints added to this rectangle and all its descendant
    rectangles are considered. If this rectangle has n descendants,
    then a list of (n+1)*4 values is returned. Every four items of the
    list contains the coordinates (x1, y1, x2, y2) of a rectangle; the
    rectangles are enumerated in depth-first order.
    """
    matrix = self.build_matrix()
    lhs, rhs = matrix[:, :-1], -matrix[:, -1]

    try:
      s = numpy.linalg.svd(lhs, compute_uv=False)
    except numpy.linalg.LinAlgError:
      raise ValueError('constraints are singular')
    if s[0]/s[-1] > 1e6:
      raise ValueError('constraints are ill-conditioned')
    try:
      return numpy.linalg.solve(lhs, rhs)
    except numpy.linalg.LinAlgError:
      raise ValueError('constraints are singular')

  def iter_vars(self):
    yield from self.vars
    for rect in self.rects:
      yield from rect.iter_vars()

  def iter_eqns(self):
    yield from self.eqns
    for rect in self.rects:
      yield from rect.iter_eqns()

  def build_matrix(self):
    vars = list(self.iter_vars())
    size = len(vars)
    matrix = numpy.zeros((size, size+1))
    vars += [Length.inch]

    i = 0
    for eqn in self.iter_eqns():
      if i == size:
        raise ValueError('too many constraints are imposed')
      for var, coeff in eqn.repr.items():
        try:
          j = vars.index(var)
        except ValueError:
          raise ValueError('length does not belong to rectangle already added')
        matrix[i, j] = coeff
      i += 1
    if i != size:
      raise ValueError('too few constraints are imposed')

    return matrix

  @staticmethod
  def ensure_list(x):
    return [x] if isinstance(x, str) or not hasattr(x, '__iter__') else x

  @staticmethod
  def get_matrix_size(matrix):
    m, n = matrix.shape
    assert m+1 == n
    return m

class BaseLayout:
  @classmethod
  def apply(cls, rect, ctxs, target=None):
    """
    Solve and apply a layout.

    This function works with matplotlib Figure and Axes. The rectangle
    rect can be regarded as a tree of rectangles. The root is rect and
    represents the Figure. Each leaf, or each rectangle containing no
    rectangles, represents an Axes. The function solves the
    constraints and applies the resulting layout to target. The
    position and size of the Figure and Axes are determined by the
    rectangle they correspond to. The function then returns a tree of
    Axes with the same structure as rect; each leaf produces an Axes,
    and any other rectangle produces a list containing Axes or other
    lists.

    The constraints solved are those added to rect and its
    descendents, complemented by a list of contexts given by ctxs.
    Contexts are constraints that are likely to change often, such as
    the overall size of the Figure. Each item of ctxs is understood as
    a constraint that the item be equal to zero.

    If target is None, a new Figure is created with the correct
    size. If target is a Figure, the Figure is resized and a
    suitable number of Axes are added. If target is a list of Axes,
    the layout is applied to the Axes directly.
    """
    from matplotlib import pyplot
    from matplotlib.figure import Figure

    ctxs = ctxs + [rect.x1, rect.y1]
    sol = cls.solve(rect, ctxs)
    (_, _, dx, dy), *sol = list(split_every(sol, 4))
    count = len(sol)

    if target is None:
      target = pyplot.figure()
    if isinstance(target, Figure):
      target = [target.add_axes([i, 0, 0, 0]) for i in range(count)]
    target = list(target)
    if not all_same(axes.figure for axes in target):
      raise ValueError('axes do not share the same figure')
    if len(target) != count:
      raise ValueError('number of rectangles does not equal number of axes')

    target[0].figure.set_size_inches(dx, dy, forward=True)
    for axes, (x1, y1, x2, y2) in zip(target, sol):
      x1, y1, x2, y2 = x1/dx, 1-y2/dy, (x2-x1)/dx, (y2-y1)/dy
      axes.set_position([x1, y1, x2, y2])

    def fill(rect, target):
      if rect.rects:
        return [fill(rect, target) for rect in rect.rects]
      else:
        return next(target)
    return fill(rect, iter(target))

  @classmethod
  def solve(cls, rect, ctxs):
    import sys
    def iter_leaf_vars(rect):
      for rect in rect.rects:
        yield from iter_leaf_vars(rect) if rect.rects else rect.vars
    vars = list(rect.iter_vars())
    index = list(range(4)) + [vars.index(x) for x in iter_leaf_vars(rect)]
    rect.eqns += ctxs
    sol = rect.solve()[index]
    rect.eqns = rect.eqns[:-len(ctxs)]
    print(cls.format_solution(sol), file=sys.stderr)
    return sol

  @staticmethod
  def format_solution(sol):
    return '\n'.join(
      '{:2} | {:7.4f} {:7.4f} {:7.4f} {:7.4f} | {:7.4f} {:7.4f}'
      .format('' if n == 0 else n-1, x1, y1, x2, y2, x2-x1, y2-y1)
      for n, (x1, y1, x2, y2) in enumerate(split_every(sol, 4)))

class PredefinedLayout(BaseLayout):
  """
  Factory for layouts with predefined spacing.
  """

  def __init__(self):
    import matplotlib
    font_size = matplotlib.rcParams['font.size']

    # padding around individual axes for labels and ticks and to maintain space
    self.top_padding                  =  Dimension.parse('0.020in') * font_size
    self.bottom_padding               =  Dimension.parse('0.045in') * font_size
    self.left_padding                 =  Dimension.parse('0.060in') * font_size
    self.right_padding                =  Dimension.parse('0.020in') * font_size
    # extra padding if axes are laid out in grid
    self.horizontal_extra_padding     = -Dimension.parse('0.020in') * font_size
    self.vertical_extra_padding       = -Dimension.parse('0.000in') * font_size
    # transverse size of colorbars
    self.colorbar_size                =  Dimension.parse('0.010in') * font_size
    # separation between axes and colorbar
    self.colorbar_sep                 =  Dimension.parse('0.000in') * font_size
    # extra padding for colorbar ticks
    self.top_colorbar_extra_padding   =  Dimension.parse('0.010in') * font_size
    self.right_colorbar_extra_padding =  Dimension.parse('0.020in') * font_size
    # figure margins
    self.top_margin                   =  Dimension.parse('0.000in') * font_size
    self.bottom_margin                =  Dimension.parse('0.000in') * font_size
    self.left_margin                  =  Dimension.parse('0.000in') * font_size
    self.right_margin                 =  Dimension.parse('0.000in') * font_size

  def make_group(self, rx, ry, cbpos='none'):
    """
    Create a new group.

    A group is a rectangle containing multiple rectangles. These
    rectangles packed tightly on a grid of len(rx) columns and
    len(ry) rows. The dx ratio among columns is rx, and the dy ratio
    among rows is ry. The caller should impose a constraint on the
    aspect ratio of one rectangle.

    If cbpos is 'right' or 'top', the group includes an additional
    rectangle to the right or the top spanning the entire grid. This
    rectangle can be used for a colorbar.
    """
    check_arg('cbpos', cbpos,  ['none', 'right', 'top'])

    nx, ny = len(rx), len(ry)
    rects = [Rectangle() for _ in range(nx*ny + (0 if cbpos == 'none' else 1))]

    result = Rectangle()
    for rect in rects:
      result.add(rect)
    result.size_to_grid(rects[:nx*ny], nx, ny)
    first, last, cbar = rects[0], rects[nx*ny-1], rects[-1]
    for i in range(1, nx):
      result.set_equal(first.dx * rx[i], rects[i].dx * rx[0])
    for i in range(1, ny):
      result.set_equal(first.dy * ry[i], rects[i*nx].dy * ry[0])

    top_padding = self.top_padding
    right_padding = self.right_padding
    if cbpos == 'right':
      result.set_equal(cbar.x1, last.x2 + self.colorbar_sep)
      result.set_equal(cbar.y1, first.y1)
      result.set_equal(cbar.dx, self.colorbar_size)
      result.set_equal(cbar.dy, last.y2 - first.y1)
      last = cbar
      right_padding = right_padding + self.right_colorbar_extra_padding
    elif cbpos == 'top':
      result.set_equal(cbar.x1, first.x1)
      result.set_equal(cbar.y2, first.y1 - self.colorbar_sep)
      result.set_equal(cbar.dx, last.x2 - first.x1)
      result.set_equal(cbar.dy, self.colorbar_size)
      first = cbar
      top_padding = top_padding + self.top_colorbar_extra_padding
    result.set_equal(result.x1, first.x1 - self.left_padding)
    result.set_equal(result.y1, first.y1 - top_padding)
    result.set_equal(result.x2, last.x2 + right_padding)
    result.set_equal(result.y2, last.y2 + self.bottom_padding)

    return result

  def make_grid(self, rects, rx, ry):
    """
    Create a new grid.

    A grid is a rectangle containing multiple rectangles. These
    rectangles are spaced out on a grid of len(rx) columns and
    len(ry) rows. The dx ratio among columns is rx, and the dy ratio
    among rows is ry. The caller should impose a constraint on the
    aspect ratio of one rectangle.

    If rects is a list of rectangles, constraints are imposed on
    them directly. If rects is a single rectangle, constraints are
    imposed on a list of len(rx)*len(ry) copies of it.
    """
    nx, ny = len(rx), len(ry)
    if isinstance(rects, Rectangle):
      rects = [rects.copy() for _ in range(nx*ny)]
    else:
      check_grid_size(rects, nx, ny)

    sx = self.horizontal_extra_padding if nx > 1 else Dimension.zero
    sy = self.vertical_extra_padding   if ny > 1 else Dimension.zero

    result = Rectangle()
    for rect in rects:
      result.add(rect)
    result.size_to_grid(rects, nx, ny, sx=sx, sy=sy)
    first, last = rects[0], rects[-1]
    for i in range(1, nx):
      result.set_equal(first.dx * rx[i], rects[i].dx * rx[0])
    for i in range(1, ny):
      result.set_equal(first.dy * ry[i], rects[i*nx].dy * ry[0])
    result.set_equal(result.x1, first.x1)
    result.set_equal(result.y1, first.y1)
    result.set_equal(result.x2, last.x2)
    result.set_equal(result.y2, last.y2)

    return result

  def make_figure(self, rect):
    """
    Create a new figure.

    A figure is a rectangle containing a single rectangle. The
    former rectangle is the entire figure, while the latter
    rectangle is the plotting area. Spacing is inserted between the
    two rectangles.
    """
    result = Rectangle()
    result.add(rect)
    result.set_equal(result.x1, rect.x1 - self.left_margin)
    result.set_equal(result.y1, rect.y1 - self.top_margin)
    result.set_equal(result.x2, rect.x2 + self.right_margin)
    result.set_equal(result.y2, rect.y2 + self.bottom_margin)
    return result

  def apply(self, rect, target=None):
    """
    Calls the superclass method with predefined contexts.
    """
    ctxs = []
    if self.figure_width is not None:
      ctxs.append(rect.dx - self.figure_width)
    if self.figure_height is not None:
      ctxs.append(rect.dy - self.figure_height)
    return super().apply(rect, ctxs, target)

class ScreenLayout(PredefinedLayout):
  """
  Factory for layouts tailored for plots shown on screen.
  """

  def __init__(self):
    super().__init__()
    self.figure_width    = Dimension.parse('6.4in')
    self.figure_height   = None
    self.bottom_padding *= 1.2
    self.left_padding   *= 1.2

class PrintLayout(PredefinedLayout):
  """
  Factory for layouts tailored for plots in publications.
  """

  def __init__(self, two_column=False):
    super().__init__()
    self.figure_width  = Dimension.parse('7.3in' if two_column else '3.475in')
    self.figure_height = None

class PresentationLayout(PredefinedLayout):
  """
  Factory for layouts tailored for plots in presentations.
  """

  def __init__(self, constraint='height'):
    super().__init__()
    if constraint == 'width':
      self.figure_width  = Dimension.parse('4in')
      self.figure_height = None
    elif constraint == 'height':
      self.figure_width  = None
      self.figure_height = Dimension.parse('3in')

if __name__ == '__main__':
  from matplotlib import pyplot
  layout = ScreenLayout()
  layout.figure_height = '4.8in'
  group = layout.make_group([1, 2, 3], [1, 2], cbpos='right')
  grid = layout.make_grid(group, [2, 1], [2, 1])
  figure = layout.make_figure(grid)
  figure = layout.apply(figure)
  for axes in pyplot.gcf().axes:
    axes.tick_params(bottom=False, top=False, left=False, right=False,
      labelcolor='none')
  for grid in figure:
    for i, (*group, _) in enumerate(grid):
      for j, axes in enumerate(group):
        pyplot.sca(axes)
        pyplot.plot([0, 1], [0, 1],
          color='C{}'.format(i), alpha=(j+1)/len(group))
  pyplot.show()
