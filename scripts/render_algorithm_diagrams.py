"""Vector explanation figures for the documented implementation, not experiments."""
from pathlib import Path
import math
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, Polygon
from reportlab.graphics import renderPDF, renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

OUT=Path('docs/assets/algorithm_v1_5')
FONT='DocumentChinese'
pdfmetrics.registerFont(TTFont(FONT,'C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
BLUE='#2563eb'; GREEN='#059669'; ORANGE='#d97706'; RED='#dc2626'; GRAY='#64748b'

def text(d,x,y,value,size=14,color='#17253b'):
    d.add(String(x,y,value,fontName=FONT,fontSize=size,fillColor=HexColor(color)))

def line(d,a,b,color=GRAY,width=1.5,dash=None):
    d.add(Line(*a,*b,strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=dash))

def arrow(d,a,b,color=GRAY,width=1.5,dash=None):
    line(d,a,b,color,width,dash)
    angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=9
    p=[b[0],b[1],b[0]-length*math.cos(angle-.4),b[1]-length*math.sin(angle-.4),
       b[0]-length*math.cos(angle+.4),b[1]-length*math.sin(angle+.4)]
    d.add(Polygon(p,fillColor=HexColor(color),strokeColor=None))

def circle(d,p,color=BLUE,r=5):
    d.add(Circle(*p,r,fillColor=HexColor(color),strokeColor=None))

def box(d,x,y,w,h,label,color=BLUE):
    d.add(Rect(x,y,w,h,rx=7,ry=7,fillColor=HexColor('#f8fafc'),strokeColor=HexColor(color),strokeWidth=1))
    text(d,x+12,y+h/2-5,label,13,color)

def base(title,subtitle,height=550):
    d=Drawing(1100,height)
    d.add(Rect(0,0,1100,height,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,32,height-40,title,24)
    text(d,32,height-69,subtitle,12,GRAY)
    return d

def export(d,name):
    OUT.mkdir(exist_ok=True,parents=True)
    pdf=OUT/f'{name}.pdf';renderPDF.drawToFile(d,str(pdf));renderSVG.drawToFile(d,str(OUT/f'{name}.svg'))
    doc=pdfium.PdfDocument(pdf);page=doc[0]
    page.render(scale=1.6).to_pil().save(OUT/f'{name}.png')
    page.close();doc.close()

def rays():
    d=base('双目射线与 GPnP 位姿约束','XZ 剖面示意；不对应实测深度或姿态，右射线已变换到左相机坐标系')
    left=(120,120);right=(375,120);observed=(505,398);predicted=(560,355)
    arrow(d,(60,95),(680,95),GRAY);text(d,675,68,'+X_L',12)
    arrow(d,(60,95),(60,425),GRAY);text(d,32,439,'+Z_L',12)
    for origin,col in [(left,BLUE),(right,GREEN)]:
        end=(origin[0]+1.12*(observed[0]-origin[0]),origin[1]+1.12*(observed[1]-origin[1]))
        arrow(d,origin,end,col,2.4)
    circle(d,left,BLUE,7);circle(d,right,GREEN,7)
    text(d,83,138,'O_L = 0',13,BLUE);text(d,326,138,'O_R = t_RL',13,GREEN)
    arrow(d,(left[0],63),(right[0],63),ORANGE);text(d,185,40,'水平基线 B > 0',13,ORANGE)
    text(d,220,285,'b_L',16,BLUE);text(d,422,300,'R_RL b_R',14,GREEN)
    circle(d,observed,GRAY,5);text(d,294,433,'理想观测射线交点',13,GRAY)
    circle(d,predicted,RED,6);text(d,558,368,'P_i = R P_T,i + t',14,RED)
    for origin,col in [(left,BLUE),(right,GREEN)]:
        vx,vy=observed[0]-origin[0],observed[1]-origin[1]
        t=((predicted[0]-origin[0])*vx+(predicted[1]-origin[1])*vy)/(vx*vx+vy*vy)
        foot=(origin[0]+t*vx,origin[1]+t*vy)
        line(d,foot,predicted,col,2,[4,3]);circle(d,foot,col,3)
    text(d,540,303,'虚线：点到两条射线的偏离',12,RED)
    box(d,735,359,328,71,'已知：模板点、像素射线、双目外参')
    box(d,735,263,328,71,'未知：旋转增量 3 维 + 平移 3 维',RED)
    arrow(d,(899,350),(899,338));arrow(d,(899,251),(899,235))
    box(d,735,154,328,71,'LM 候选 → 数值状态 → 几何验收',GREEN)
    text(d,737,117,'e_L = b_L × P_i',14,BLUE)
    text(d,737,85,'e_R = (R_RL b_R) × (P_i − t_RL)',13,GREEN)
    text(d,32,16,'模板原点与局部回退模型仍需区分；输出为模板到左相机的 R、t。仅视差实验不执行此步骤。',12,GRAY)
    export(d,'gpnp_rays')

def optical_flow():
    d=base('极线带、右点初始化与光流往返校验','示意数值，非实验数据；允许垂直偏差不会消除标定误差。坐标间距未按比例绘制。')
    for x,title in [(40,'左图：仅在左 ROI 提取特征'),(606,'右图：在灰度搜索区域寻找对应')]:
        d.add(Rect(x,243,450,190,fillColor=HexColor('#f8fafc'),strokeColor=HexColor('#cbd5e1')))
        text(d,x+8,444,title,14)
        arrow(d,(x+15,419),(x+90,419));text(d,x+98,415,'u',12)
        arrow(d,(x+15,419),(x+15,363));text(d,x+5,348,'v',12)
    d.add(Rect(657,287,350,72,fillColor=HexColor('#dbeafe'),strokeColor=None))
    line(d,(657,323),(1007,323),BLUE,1,[4,3]);text(d,730,379,'极线 ±3 px 搜索带',13,BLUE)
    p=(300,323);q=(794,305);seed=(834,323)
    circle(d,p,BLUE);circle(d,q,GREEN);circle(d,seed,ORANGE)
    text(d,190,280,'p_L=(300,200)',14,BLUE)
    text(d,672,263,'p_R=(260,201.5)',14,GREEN)
    text(d,839,335,'同坐标初值',12,ORANGE)
    arrow(d,seed,q,ORANGE,2);arrow(d,p,q,GREEN,1.6,[5,4])
    text(d,458,359,'L→R LK',13,GREEN)
    arrow(d,(q[0],244),(p[0],244),ORANGE,1.8,[5,4]);text(d,452,223,'R→L 反向追踪',13,ORANGE)
    text(d,70,192,'返回点 (300.4,200.3)：FB=0.5 px；水平视差 d=40 px；垂直偏差 1.5 px',15)
    labels=['左角点与原编号','右初值／粗搜索','L→R LK','R→L、NCC等质检','可用平面与帧级检查']
    for i,label in enumerate(labels):
        x=32+i*216;box(d,x,97,200,55,label,GREEN if i==4 else BLUE)
        if i<4:arrow(d,(x+200,125),(x+214,125))
    text(d,34,56,'逐点合格后仍可能有残余错误；平面过滤只在开关开启、冗余足够且共识可靠时剔除超限点。',12,GRAY)
    text(d,34,27,'四点 TT 跳过平面过滤；最终右点必须来自图像观测，投影或粗搜索点不能直接作为最终对应。',12,GRAY)
    export(d,'optical_flow_epipolar')

def identity():
    d=base('平面过滤保留身份，禁止按剩余点数重新配模板','示意剔除中间编号 2；本次实际第 29 帧剔除的是编号 9。示意不是实验测量。',480)
    text(d,35,355,'逐点质检后',15)
    for i in range(10):box(d,220+i*75,331,57,48,str(i),RED if i==2 else BLUE)
    arrow(d,(523,321),(523,277),RED);text(d,555,294,'可靠平面：剔除超限点 2',14,RED)
    text(d,35,233,'几何过滤后',15)
    for j,i in enumerate([0,1,3,4,5,6,7,8,9]):box(d,220+j*75,208,57,48,str(i),GREEN)
    text(d,222,172,'图像点／右点／视差／模板身份按同一原编号同步筛选',14,GREEN)
    box(d,35,87,300,52,'关闭开关或不足 6 点：跳过',GRAY)
    box(d,390,87,300,52,'无可靠共识：标记不确定，保留',ORANGE)
    box(d,745,87,320,52,'过滤后复核点数、比例与空间覆盖',BLUE)
    text(d,36,45,'倾斜平面允许 d = a·u + b·v + c；几何残差不是对真值误差直接截断。',14)
    text(d,36,19,'满足数值与几何验收的点才能进入后续位姿约束；TT 仍要求原四点全部匹配通过。',12,GRAY)
    export(d,'planar_identity_filter')

if __name__=='__main__':
    rays();optical_flow();identity();print('Rendered three algorithm diagrams as PNG, SVG and PDF')
