import altair as alt
from vega_datasets import data
import pandas as pd


def plot_tsne(data, darkmode=True):

    # import IPython # Try to automatically detect darkmode - colab is blocking my DOM request
    # # html[theme=dark]
    # js_code = r'document.documentElement.getAttribute("theme");'
    # display(IPython.display.Javascript(js_code))

    alt.renderers.set_embed_options(theme='dark' if darkmode else 'light')
    selection = alt.selection_single(on='mouseover', clear='false', nearest=True,
                                    init={'x':data['x'][0], 'y':data['y'][0]})
    scatter = alt.Chart(data).mark_circle().encode(
        alt.X('x:N',axis=None),
        alt.Y('y:N',axis=None),
        color=alt.condition(selection,
                            alt.value('lightgray'),
                            alt.Color('label:N')),
        # shape= alt.Shape('label:N', condition=selection,scale=alt.Scale(range=['circle','diamond'])), 
        size=alt.value(100),
        tooltip='label:N'
    ).add_selection(
        selection
    ).properties(
        width=400,
        height=400
    )

    digit  = alt.Chart(data).transform_filter(
        selection
    ).transform_window(
        index='count()'           # number each of the images
    ).transform_flatten(
        ['image']                 # extract rows from each image
    ).transform_window(
        row='count()',            # number the rows...
        groupby=['index']         # ...within each image
    ).transform_flatten(
        ['image']                 # extract the values from each row
    ).transform_window(
        column='count()',         # number the columns...
        groupby=['index', 'row']  # ...within each row & image
    ).mark_rect(stroke='black',strokeWidth=0).encode(
        alt.X('column:O', axis=None),
        alt.Y('row:O', axis=None),
        alt.Color('image:Q',
            scale=alt.Scale(scheme=alt.SchemeParams('greys', extent=[1, 0])),
            legend=None
        ),
    ).properties(
        width=400,
        height=400,
    )

    return scatter | digit

