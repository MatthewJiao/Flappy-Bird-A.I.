import pygame #done
import neat
#import time
import os
import random
pygame.font.init()
WIN_WIDTH = 500
WIN_HEIGHT = 700
COUNTPRIME = 10
GEN = 0
MAXSCORE = 0
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load("bird1.png")), pygame.transform.scale2x(pygame.image.load("bird2.png")), pygame.transform.scale2x(pygame.image.load("bird3.png"))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load("pipe.png"))
BASE_IMG = pygame.transform.scale2x(pygame.image.load("base.png"))
BG_IMG = pygame.transform.scale2x(pygame.image.load("bg.png"))

STAT_FONT = pygame.font.SysFont("comicsans", 20)
STAT_FONT2 = pygame.font.SysFont("comicsans", 70)


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5 #S/E is positive
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel*self.tick_count + 1.5*self.tick_count**2 # vertical dist
        #print(d)
        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4+1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self): #maybe just use squares?
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200 #gap of 150 is possible
    VEL = 10

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(60, 340)
        self.top = self.height - self.PIPE_TOP.get_height() #west/north corner
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL
        #print(self.x)

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset) # collision
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False


class Base:
    VEL = 10 #
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH #if increase refresh rate, replace x2 with 0
            #print("X2 ", self.x2)
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
        #print("x1 ", self.x1)
        #print("x2 ", self.x2)


def draw_window(win, birds, pipes, base, score, gen, count, maxScore):
    win.blit(BG_IMG, (-50,-200))
    for pipe in pipes:
        pipe.draw(win)
    text = STAT_FONT2.render(str(score), 1, (255,255,255))
    win.blit(text, (235, 20)) #WIN_WIDTH - 10 - text.get_width()

    text = STAT_FONT.render("Generation: " + str(gen), 1, (255,255,255))
    win.blit(text, (10, 10))

    text = STAT_FONT.render("Status: " + str(count), 1, (255,255,255))
    win.blit(text, (10, 30))

    text = STAT_FONT.render("Max_Score: " + str(maxScore), 1, (255,255,255))
    win.blit(text, (10, 50))

    base.draw(win)

    for bird in birds:
       bird.draw(win)
    pygame.display.update()


def main(genomes, config): #testing for fitness
    global GEN
    global COUNTPRIME
    global MAXSCORE
    GEN += 1
    count = COUNTPRIME

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(600)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        #bird.move()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break


        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    count -= 1
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score +=1
            if score>MAXSCORE:
                MAXSCORE = score
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))
        for r in rem:
            pipes.remove(r)
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 600 or bird.y < 0:
                #ge[x].fitness += 100
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
                count -= 1

       # if score > 3:
       #     break
        base.move()
        draw_window(win, birds, pipes, base, score, GEN, count, MAXSCORE)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,1000) #50 fitness threshold

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
