"""Stable analytic 2D character rig and accessory motion."""
import math
from PIL import ImageDraw
from motion_core import C, clamp, smooth, spring, lpoint, rotate_points, round_line, rounded, line, lerp

class CharacterRig:
    """One simple 2D human rig: head/eye gaze, torso, two arms, two legs, hair and a gravity-hung bag."""
    def __init__(self):
        self.skin=C['skin']; self.dark=C['hair']
    def _ik(self,shoulder,target,l1=136,l2=126,side=1):
        dx,dy=target[0]-shoulder[0],target[1]-shoulder[1]; raw=math.hypot(dx,dy) or 1
        dist=clamp(raw,abs(l1-l2)+2,l1+l2-2); ux,uy=dx/raw,dy/raw
        along=(l1*l1-l2*l2+dist*dist)/(2*dist); perp=math.sqrt(max(0,l1*l1-along*along))
        ex=shoulder[0]+along*ux+side*(-uy)*perp; ey=shoulder[1]+along*uy+side*ux*perp
        return (ex,ey)
    def draw(self,im,t,target,gesture,settle=0.0):
        d=ImageDraw.Draw(im,'RGBA')
        # Enter with a short walk, decelerate smoothly, then settle into stable readable poses.
        walk_left=1-smooth((t-.15)/2.6)
        root_x=lerp(1635,2020,1.0)-0 if t<.15 else lerp(1635,2020,0) # fixed target after easing below
        x=2025+(1635-2025)*smooth((t-.12)/2.30)
        if t<.12: x=2025
        if t>2.42: x=1635+2.5*math.sin(t*0.65)
        phase=t*7.4
        step_amp=walk_left*(1-smooth((t-.05)/2.55))
        bob=-abs(math.sin(phase))*8*step_amp
        torso_sway=math.sin(t*1.6)*2.5 + spring(t-2.40,2.4,4.2,8.5)
        ground=908+bob
        x+=torso_sway
        # All joint locations share one rig coordinate frame and deterministic motion curves.
        head_spring=sum(spring(t-start,amp,3.1,9.0) for start,amp in [(2.3,4.0),(6.35,2.3),(11.0,2.7),(17.75,3.0),(22.9,2.6),(30.45,3.1),(37.25,2.4)])
        head_x=x-2+head_spring*.44
        head_y=ground-500+ bob*.24 + 1.6*math.sin(t*1.4) + head_spring*.22
        shL=(x-49,ground-409+torso_sway); shR=(x+48,ground-409-torso_sway*.4)
        hipL=(x-28,ground-226); hipR=(x+29,ground-226)
        walk=step_amp
        # legs swing at hips; knees lag via a second harmonic; the feet land on the same ground.
        idle_settle=smooth((t-2.20)/.75)
        for side,hip,offset,col in [(-1,hipL,0,C['pants']),(1,hipR,math.pi,C['pants'])]:
            a=phase+offset; swing=math.sin(a)*36*walk; knee_bend=max(0,math.cos(a))*32*walk
            weight_shift=math.sin(t*1.22+offset)*3.2*idle_settle
            knee=(hip[0]+swing*.55+weight_shift*.42,ground-122+knee_bend*.32+max(0,math.sin(t*1.22+offset))*2.0*idle_settle)
            ankle=(hip[0]+swing+weight_shift,ground-28+max(0,math.sin(a))*8*walk)
            round_line(d,hip,knee,(18,26,39),38); round_line(d,hip,knee,col,29)
            round_line(d,knee,ankle,(18,26,39),31); round_line(d,knee,ankle,col,23)
            # subtle trouser seam, aligned to the current limb segment
            p1=lpoint(knee,ankle,.24); p2=lpoint(knee,ankle,.75); line(d,p1,p2,(88,105,126,190),2)
            footx=ankle[0]+13; footy=ground-12+max(0,math.sin(a))*8*walk
            rounded(d,(footx-14,footy-8,footx+35,footy+11),8,C['shoe'],(8,12,17),2)
            d.ellipse((int(footx-10),int(footy+4),int(footx+31),int(footy+14)),fill=(0,0,0,45))
        # rear arm swings while the figure walks; all upper/lower sections have their own joints.
        back_wrist=(x+88+math.sin(phase+math.pi)*28*walk,ground-288+math.cos(phase)*24*walk)
        back_elbow=(x+76+math.sin(phase+math.pi)*10*walk,ground-340)
        round_line(d,shR,back_elbow,(20,45,54),35); round_line(d,shR,back_elbow,C['shirt'],27)
        round_line(d,back_elbow,back_wrist,(20,45,54),28); round_line(d,back_elbow,back_wrist,self.skin,20)
        d.ellipse((back_wrist[0]-12,back_wrist[1]-13,back_wrist[0]+12,back_wrist[1]+13),fill=self.skin)
        # torso with shoulder slope and waist taper; the sling strap is drawn above it with correct endpoints.
        torso=[(shL[0]-5,shL[1]+3),(shR[0]+5,shR[1]+3),(x+61,ground-242),(x-59,ground-242)]
        d.polygon(torso,fill=C['shirt'],outline=(16,61,69),width=4)
        d.polygon([(x-37,ground-395),(x+10,ground-393),(x+43,ground-245),(x-10,ground-245)],fill=(50,145,139,130))
        # folded hem and two fabric seams move with the torso, not the screen.
        line(d,(x-42,ground-255),(x+43,ground-255),(27,82,86),3)
        line(d,(x-15,ground-371),(x-25,ground-293),(110,186,171,150),2)
        # Cross-body strap attaches shoulder to opposite hip. The small bag hangs under gravity.
        strap_start=(x-28,ground-401); strap_end=(x+43,ground-242)
        round_line(d,strap_start,strap_end,(19,45,50),13); round_line(d,strap_start,strap_end,C['gold'],7)
        bag_ang=.08*math.sin(t*2.1)+spring(t-2.3,.11,3.2,8.8)+walk*.10*math.sin(phase+1.1)
        bag_c=(x+49+4*math.sin(t*2.0)*walk,ground-250+spring(t-2.3,5,3.3,8.4))
        bagpts=rotate_points([(-39,-25),(39,-25),(44,27),(-44,27)],bag_c[0],bag_c[1],bag_ang)
        d.polygon(bagpts,fill=(46,67,75),outline=C['gold'],width=3)
        d.line(rotate_points([(-27,-5),(26,-5)],bag_c[0],bag_c[1],bag_ang),fill=(123,159,158),width=3)
        # Left/forward arm uses two-bone inverse kinematics to point at the active teaching object.
        if gesture=='point':
            # Tiny deterministic reach-and-settle tremor gives the hand life without jitter.
            wrist=(target[0]+1.8*math.sin(t*1.7),target[1]+1.6*math.sin(t*1.3+0.4)); elbow=self._ik(shL,wrist,142,132,side=1)
            round_line(d,shL,elbow,(18,50,57),37); round_line(d,shL,elbow,C['shirtlight'],28)
            round_line(d,elbow,wrist,(18,50,57),28); round_line(d,elbow,wrist,self.skin,20)
            d.ellipse((wrist[0]-13,wrist[1]-13,wrist[0]+13,wrist[1]+13),fill=self.skin,outline=C['skinshadow'],width=2)
            # index finger extends toward the selected formula; remaining fingers curl back
            fingertip=(wrist[0]-24,wrist[1]-7)
            round_line(d,(wrist[0]-1,wrist[1]-2),fingertip,self.skin,9)
            for k in range(3):
                round_line(d,(wrist[0]-1,wrist[1]+4+k*2),(wrist[0]+8,wrist[1]+11+k*4),self.skin,6)
        elif gesture=='open':
            wrist=(x-152,ground-328); elbow=(x-105,ground-359)
            round_line(d,shL,elbow,(18,50,57),37); round_line(d,shL,elbow,C['shirtlight'],28)
            round_line(d,elbow,wrist,(18,50,57),28); round_line(d,elbow,wrist,self.skin,20)
            d.ellipse((wrist[0]-13,wrist[1]-13,wrist[0]+13,wrist[1]+13),fill=self.skin)
            for k in range(4): round_line(d,(wrist[0]-4+k*2,wrist[1]-1),(wrist[0]-12+k*8,wrist[1]-20-k%2*4),self.skin,5)
        else:
            wrist=(x-94,ground-279); elbow=(x-91,ground-338)
            round_line(d,shL,elbow,(18,50,57),37); round_line(d,shL,elbow,C['shirtlight'],28)
            round_line(d,elbow,wrist,(18,50,57),28); round_line(d,elbow,wrist,self.skin,20)
            d.ellipse((wrist[0]-12,wrist[1]-13,wrist[0]+12,wrist[1]+13),fill=self.skin)
        # neck and head are anchored to shoulder rig; no nose or mouth, eyes only.
        rounded(d,(head_x-17,head_y+44,head_x+17,head_y+86),10,self.skin)
        d.ellipse((head_x-49,head_y-63,head_x+49,head_y+65),fill=self.skin,outline=C['skinshadow'],width=3)
        # Hair cap follows the head anchor; individual locks are separate and lag through a damped spring.
        hs=head_spring*.60 + .7*math.sin(t*2.4)
        cap=[(head_x-48,head_y-18),(head_x-47,head_y-47),(head_x-33,head_y-62),(head_x-10,head_y-68),(head_x+18,head_y-62),(head_x+42,head_y-44),(head_x+45,head_y-23),(head_x+27,head_y-31),(head_x+8,head_y-39),(head_x-12,head_y-29),(head_x-29,head_y-34)]
        d.polygon(cap,fill=self.dark,outline=(24,22,22),width=2)
        for j in range(4):
            ax=head_x-34+j*21; ay=head_y-53+(j%2)*7; tipx=ax+12+hs*(.45+j*.08); tipy=ay-13+hs*.27
            d.polygon([(ax-7,ay+9),(ax+4,ay-7),(tipx,tipy),(ax+15,ay+7)],fill=self.dark)
        # Hair side lock falls beside temple with a tiny gravity lag; avoids floating away from scalp.
        d.polygon([(head_x+32,head_y-35),(head_x+45,head_y-18),(head_x+40,head_y+10+hs*.20),(head_x+29,head_y+22),(head_x+31,head_y-7)],fill=self.dark)
        # eyes are the only facial marks; pupils track current focus and blink briefly.
        blink=min(1.0, max(0.0,(.08-abs((t%4.7)-.05))/.08))
        gy= head_y+3
        for ex in (head_x-17,head_x+17):
            if blink>.8:
                line(d,(ex-6,gy),(ex+6,gy),(38,30,29),3)
            else:
                d.ellipse((ex-6,gy-5,ex+6,gy+7),fill=(252,242,220),outline=(67,48,42),width=1)
                gaze=2.2 if target[0]>head_x else -2.2
                d.ellipse((ex-2+gaze,gy-1,ex+3+gaze,gy+5),fill=(30,29,28))
