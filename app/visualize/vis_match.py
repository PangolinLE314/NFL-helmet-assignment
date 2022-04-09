import plotly.express as px
import plotly.graph_objects as go


def add_plotly_field(fig):
    # Reference https://www.kaggle.com/ammarnassanalhajali/nfl-big-data-bowl-2021-animating-players
    fig.update_traces(marker_size=20)

    fig.update_layout(paper_bgcolor='#29a500', plot_bgcolor='#29a500', font_color='white',
                      width=800,
                      height=600,
                      title="",

                      xaxis=dict(
                          nticks=10,
                          title="",
                          visible=False
                      ),

                      yaxis=dict(
                          scaleanchor="x",
                          title="Temp",
                          visible=False
                      ),
                      showlegend=True,

                      annotations=[
                          dict(
                              x=-5,
                              y=26.65,
                              xref="x",
                              yref="y",
                              text="ENDZONE",
                              font=dict(size=16, color="#e9ece7"),
                              align='center',
                              showarrow=False,
                              yanchor='middle',
                              textangle=-90
                          ),
                          dict(
                              x=105,
                              y=26.65,
                              xref="x",
                              yref="y",
                              text="ENDZONE",
                              font=dict(size=16, color="#e9ece7"),
                              align='center',
                              showarrow=False,
                              yanchor='middle',
                              textangle=90
                          )]
                      ,
                      legend=dict(
                          traceorder="normal",
                          font=dict(family="sans-serif", size=12),
                          title="",
                          orientation="h",
                          yanchor="bottom",
                          y=1.00,
                          xanchor="center",
                          x=0.5
                      ),
                      )
    ####################################################

    fig.add_shape(type="rect", x0=-10, x1=0, y0=0, y1=53.3, line=dict(color="#c8ddc0", width=3), fillcolor="#217b00",
                  layer="below")
    fig.add_shape(type="rect", x0=100, x1=110, y0=0, y1=53.3, line=dict(color="#c8ddc0", width=3), fillcolor="#217b00",
                  layer="below")
    for x in range(0, 100, 10):
        fig.add_shape(type="rect", x0=x, x1=x + 10, y0=0, y1=53.3, line=dict(color="#c8ddc0", width=3),
                      fillcolor="#29a500", layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=1, x1=x, y1=2, line=dict(color="#c8ddc0", width=2), layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=51.3, x1=x, y1=52.3, line=dict(color="#c8ddc0", width=2), layer="below")

    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=20.0, x1=x, y1=21, line=dict(color="#c8ddc0", width=2), layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=32.3, x1=x, y1=33.3, line=dict(color="#c8ddc0", width=2), layer="below")

    fig.add_trace(go.Scatter(
        x=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98], y=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        text=["G", "1 0", "2 0", "3 0", "4 0", "5 0", "4 0", "3 0", "2 0", "1 0", "G"],
        mode="text",
        textfont=dict(size=20, family="Arail"),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98],
        y=[48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3],
        text=["G", "1 0", "2 0", "3 0", "4 0", "5 0", "4 0", "3 0", "2 0", "1 0", "G"],
        mode="text",
        textfont=dict(size=20, family="Arail"),
        showlegend=False,
    ))

    return fig


def vis_video(tr_tracking, game_play, html_out=None, write=True):
    """
    Visualize video using plotly
    :param tr_tracking:
    :param game_play:
    :param html_out:
    :param write:
    :return:
    """
    fig = px.scatter(
        tr_tracking.query("game_play == @game_play"),
        x="x",
        y="y",
        range_x=[-10, 110],
        range_y=[-10, 53.3],
        hover_data=["player", "s", "a", "dir"],
        color="team",
        animation_frame="track_time_count",
        text="player",
        title=f"Animation of NGS data for game_play {game_play}",
    )

    fig.update_traces(textfont_size=10)
    fig = add_plotly_field(fig)
    if write:
        fig.write_html(html_out)
    return fig


import os
import cv2
import subprocess
import pandas as pd


def video_with_baseline_boxes(
        video_path: str, baseline_boxes: pd.DataFrame, gt_labels: pd.DataFrame, verbose=True
) -> str:
    """
    Annotates a video with both the baseline model boxes and ground truth boxes.
    Baseline model prediction confidence is also displayed.
    """
    VIDEO_CODEC = "MP4V"
    HELMET_COLOR = (0, 0, 0)  # Black
    BASELINE_COLOR = (255, 255, 255)  # White
    IMPACT_COLOR = (0, 0, 255)  # Red
    video_name = os.path.basename(video_path).replace(".mp4", "")
    if verbose:
        print(f"Running for {video_name}")
    baseline_boxes = baseline_boxes.copy()
    gt_labels = gt_labels.copy()

    baseline_boxes["video"] = (
        baseline_boxes["video_frame"].str.split("_").str[:3].str.join("_")
    )
    gt_labels["video"] = gt_labels["video_frame"].str.split("_").str[:3].str.join("_")
    baseline_boxes["frame"] = (
        baseline_boxes["video_frame"].str.split("_").str[-1].astype("int")
    )
    gt_labels["frame"] = gt_labels["video_frame"].str.split("_").str[-1].astype("int")

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "labeled_" + video_name + ".mp4"
    tmp_output_path = "tmp_" + output_path
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height)
    )
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        frame += 1

        # Let's add a frame index to the video so we can track where we are
        img_name = f"{video_name}_frame{frame}"
        cv2.putText(
            img,
            img_name,
            (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            HELMET_COLOR,
            thickness=2,
        )

        # Now, add the boxes
        boxes = baseline_boxes.query("video == @video_name and frame == @frame")
        if len(boxes) == 0:
            print("Boxes incorrect")
            return
        for box in boxes.itertuples(index=False):
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                BASELINE_COLOR,
                thickness=1,
            )
            cv2.putText(
                img,
                f"{box.conf:0.2}",
                (box.left, max(0, box.top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BASELINE_COLOR,
                thickness=1,
            )

        boxes = gt_labels.query("video == @video_name and frame == @frame")
        if len(boxes) == 0:
            print("Boxes incorrect")
            return
        for box in boxes.itertuples(index=False):
            # Filter for definitive head impacts and turn labels red
            if box.isDefinitiveImpact == True:
                color, thickness = IMPACT_COLOR, 3
            else:
                color, thickness = HELMET_COLOR, 1
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                color,
                thickness=thickness,
            )
            cv2.putText(
                img,
                box.label,
                (box.left + 1, max(0, box.top - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=1,
            )

        output_video.write(img)
    output_video.release()
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            tmp_output_path,
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-vcodec",
            "libx264",
            output_path,
        ]
    )
    os.remove(tmp_output_path)

    return output_path
