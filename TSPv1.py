import random
import sys
import getopt
from PIL import Image, ImageDraw, ImageFont
import math
from math import sqrt
import logging


def hillclimb(init_function,move_operator,objective_function,max_evaluations):
    '''
    hillclimb until either max_evaluations is reached or we are at a local optima
    '''
    best=init_function()
    best_score=objective_function(best)
    
    num_evaluations=1
    
    logging.info('hillclimb started: score=%f',best_score)
    
    while num_evaluations < max_evaluations:
        # examine moves around our current position
        move_made=False
        for next in move_operator(best):
            if num_evaluations >= max_evaluations:
                break
            
            # see if this move is better than the current
            next_score=objective_function(next)
            num_evaluations+=1
            if next_score > best_score:
                best=next
                best_score=next_score
                move_made=True
                break # depth first search
            
        if not move_made:
            break # we couldn't find a better move (must be at a local maximum)
    
    logging.info('hillclimb finished: num_evaluations=%d, best_score=%f',num_evaluations,best_score)
    return (num_evaluations,best_score,best)

def hillclimb_and_restart(init_function,move_operator,objective_function,max_evaluations):
    '''
    repeatedly hillclimb until max_evaluations is reached
    '''
    best=None
    best_score=0
    
    num_evaluations=0
    while num_evaluations < max_evaluations:
        remaining_evaluations=max_evaluations-num_evaluations
        
        logging.info('(re)starting hillclimb %d/%d remaining',remaining_evaluations,max_evaluations)
        evaluated,score,found=hillclimb(init_function,move_operator,objective_function,remaining_evaluations)
        
        num_evaluations+=evaluated
        if score > best_score or best is None:
            best_score=score
            best=found
        
    return (num_evaluations,best_score,best)


# Simulated annealing ---------------------------------------------------------
def P(prev_score, next_score, temperature):
    if next_score > prev_score:
        return 1.0
    else:
        return math.exp(-abs(next_score-prev_score)/temperature )

class ObjectiveFunction:
    '''class to wrap an objective function and 
    keep track of the best solution evaluated'''
    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.best=None
        self.best_score=None
    
    def __call__(self, solution):
        score = self.objective_function(solution)
        if self.best is None or score > self.best_score:
            self.best_score=score
            self.best=solution
            logging.info('new best score: %f',self.best_score)
        return score

def kirkpatrick_cooling(start_temp,alpha):
    T=start_temp
    while True:
        yield T
        T=alpha*T

def anneal(init_function,move_operator,objective_function,max_evaluations,start_temp,alpha):
    
    # wrap the objective function (so we record the best)
    objective_function=ObjectiveFunction(objective_function)
    
    current=init_function()
    current_score=objective_function(current)
    num_evaluations=1
    
    cooling_schedule=kirkpatrick_cooling(start_temp,alpha)
    
    logging.info('anneal started: score=%f',current_score)
    
    for temperature in cooling_schedule:
        done = False
        # examine moves around our current position
        for next in move_operator(current):
            if num_evaluations >= max_evaluations:
                done=True
                break
            
            next_score=objective_function(next)
            num_evaluations+=1
            
            # probablistically accept this solution
            # always accepting better solutions
            p=P(current_score,next_score,temperature)
            if random.random() < p:
                current=next
                current_score=next_score
                break
        # see if completely finished
        if done: break
    
    best_score=objective_function.best_score
    best=objective_function.best
    logging.info('final temperature: %f',temperature)
    logging.info('anneal finished: num_evaluations=%d, best_score=%f',num_evaluations,best_score)
    return (num_evaluations,best_score,best)
    

# TSP -------------------------------------------------------------------------
def rand_seq(size):
    '''generates values in random order
    equivalent to using shuffle in random,
    without generating all values at once'''
    values=range(size)
    for i in range(size):
        # pick a random index into remaining values
        j=i+int(random.random()*(size-i))
        # swap the values
        values[j], values[i] = values[i], values[j]
        # return the swapped value
        yield values[i] 

def all_pairs(size):
    '''generates all i,j pairs for i,j from 0-size'''
    for i in rand_seq(size):
        for j in rand_seq(size):
            yield (i,j)

def reversed_sections(tour):
    '''generator to return all possible variations where the section between two cities are swapped'''
    for i,j in all_pairs(len(tour)):
        if i != j:
            copy=tour[:]
            if i < j:
                copy[i:j+1]=reversed(tour[i:j+1])
            else:
                copy[i+1:]=reversed(tour[:j])
                copy[:j]=reversed(tour[i+1:])
            if copy != tour: # no point returning the same tour
                yield copy

def swapped_cities(tour):
    '''generator to create all possible variations where two cities have been swapped'''
    for i,j in all_pairs(len(tour)):
        if i < j:
            copy=tour[:]
            copy[i], copy[j]=tour[j], tour[i]
            yield copy

def cartesian_matrix(coords):
    '''create a distance matrix for the city coords that uses straight line distance'''
    matrix={}
    for i,(x1,y1) in enumerate(coords):
        for j,(x2,y2) in enumerate(coords):
            dx,dy=x1-x2,y1-y2
            dist=sqrt(dx*dx + dy*dy)
            matrix[i,j]=dist
    return matrix

def read_coords(coord_file):
    '''
    read the coordinates from file and return the distance matrix.
    coords should be stored as comma separated floats, one x,y pair per line.
    '''
    coords=[]
    for line in coord_file:
        x,y=line.strip().split(',')
        coords.append((float(x),float(y)))
    return coords

def tour_length(matrix,tour):
    '''total up the total length of the tour based on the distance matrix'''
    total=0
    num_cities=len(tour)
    for i in range(num_cities):
        j=(i+1)%num_cities
        city_i=tour[i]
        city_j=tour[j]
        total+=matrix[city_i,city_j]
    return total

def write_tour_to_img(coords,tour,title,img_file):
    padding=20
    # shift all coords in a bit
    coords=[(x+padding,y+padding) for (x,y) in coords]
    maxx,maxy=0,0
    for x,y in coords:
        maxx=max(x,maxx)
        maxy=max(y,maxy)
    maxx+=padding
    maxy+=padding
    img=Image.new("RGB",(int(maxx),int(maxy)),color=(255,255,255))
    
    font=ImageFont.load_default()
    d=ImageDraw.Draw(img);
    num_cities=len(tour)
    for i in range(num_cities):
        j=(i+1)%num_cities
        city_i=tour[i]
        city_j=tour[j]
        x1,y1=coords[city_i]
        x2,y2=coords[city_j]
        d.line((int(x1),int(y1),int(x2),int(y2)),fill=(0,0,0))
        d.text((int(x1)+7,int(y1)-5),str(i),font=font,fill=(32,32,32))
    
    
    for x,y in coords:
        x,y=int(x),int(y)
        d.ellipse((x-5,y-5,x+5,y+5),outline=(0,0,0),fill=(196,196,196))
    
    d.text((1,1),title,font=font,fill=(0,0,0))
    
    del d
    img.save(img_file, "PNG")

def init_random_tour(tour_length):
   tour=range(tour_length)
   random.shuffle(tour)
   return tour

def run_hillclimb(init_function,move_operator,objective_function,max_iterations):
    from hillclimb import hillclimb_and_restart
    iterations,score,best=hillclimb_and_restart(init_function,move_operator,objective_function,max_iterations)
    return iterations,score,best

def run_anneal(init_function,move_operator,objective_function,max_iterations,start_temp,alpha):
    if start_temp is None or alpha is None:
        usage();
        print("missing --cooling start_temp:alpha for annealing")
        sys.exit(1)
    from sa import anneal
    iterations,score,best=anneal(init_function,move_operator,objective_function,max_iterations,start_temp,alpha)
    return iterations,score,best

def usage():
    print("usage: python %s [-o <output image file>] [-v] [-m reversed_sections|swapped_cities] -n <max iterations> [-a hillclimb|anneal] [--cooling start_temp:alpha] <city file>" % sys.argv[0])

def main():
    try:
        options, args = getopt.getopt(sys.argv[1:], "ho:vm:n:a:", ["cooling="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    out_file_name=None
    max_iterations=None
    verbose=None
    move_operator=reversed_sections
    run_algorithm=run_hillclimb
    
    start_temp,alpha=None,None
    
    for option,arg in options:
        if option == '-v':
            verbose=True
        elif option == '-h':
            usage()
            sys.exit()
        elif option == '-o':
            out_file_name=arg
        elif option == '-n':
            max_iterations=int(arg)
        elif option == '-m':
            if arg == 'swapped_cities':
                move_operator=swapped_cities
            elif arg == 'reversed_sections':
                move_operator=reversed_sections
        elif option == '-a':
            if arg == 'hillclimb':
                run_algorithm=run_hillclimb
            elif arg == 'anneal':
                # do this to pass start_temp and alpha to run_anneal
                def run_anneal_with_temp(init_function,move_operator,objective_function,max_iterations):
                    return run_anneal(init_function,move_operator,objective_function,max_iterations,start_temp,alpha)
                run_algorithm=run_anneal_with_temp
        elif option == '--cooling':
            start_temp,alpha=arg.split(':')
            start_temp,alpha=float(start_temp),float(alpha)
    
    if max_iterations is None:
        usage();
        sys.exit(2)
    
    if out_file_name and not out_file_name.endswith(".png"):
        usage()
        print("output image file name must end in .png")
        sys.exit(1)
    
    if len(args) != 1:
        usage()
        print("no city file specified")
        sys.exit(1)
    
    city_file=args[0]
    
    # enable more verbose logging (if required) so we can see workings
    # of the algorithms
    import logging
    format='%(asctime)s %(levelname)s %(message)s'
    if verbose:
        logging.basicConfig(level=logging.INFO,format=format)
    else:
        logging.basicConfig(format=format)
    
    # setup the things tsp specific parts hillclimb needs
    coords=read_coords(file(city_file))
    init_function=lambda: init_random_tour(len(coords))
    matrix=cartesian_matrix(coords)
    objective_function=lambda tour: -tour_length(matrix,tour)
    
    logging.info('using move_operator: %s'%move_operator)
    
    iterations,score,best=run_algorithm(init_function,move_operator,objective_function,max_iterations)
    # output results
    print(iterations,score,best)
    
    if out_file_name:
        write_tour_to_img(coords,best,'%s: %f'%(city_file,score),file(out_file_name,'w'))

if __name__ == "__main__":
    main()
    
    
# Test Hillclimb --------------------------------------------------------------
def objective_function(i):
    return i

max_evaluations=500

def test_simple_hillclimb():
    '''
    test whether given a really simple move
    operator that just increments the number given
    we always end up with with largest number
    we can get after the number of evaluations
    we specify
    '''
    def move_operator(i):
        yield i+1
    def init_function():
        return 1
    
    num_evaluations,best_score,best=hillclimb.hillclimb(init_function,move_operator,objective_function,max_evaluations)
    
    assert num_evaluations == max_evaluations
    assert best == max_evaluations
    assert best_score == max_evaluations

def test_peak_hillclimb():
    '''
    check we hit the peak value (and don't iterate more than we need to)
    '''
    def move_operator(i):
        if i < 100:
            yield i+1
    def init_function():
        return 1
    
    
    num_evaluations,best_score,best=hillclimb.hillclimb(init_function,move_operator,objective_function,max_evaluations)
    
    assert num_evaluations <= max_evaluations
    assert num_evaluations == 100
    assert best == 100
    assert best_score == 100

def test_hillclimb_and_restart():
    '''
    see whether we restart on each number correctly
    '''
    def move_operator(i):
        # even numbers only go up to 50
        if i % 2 == 0 and i < 50:
            yield i+2
        elif i % 2 != 0 and i < 100: # odd numbers go higher
            yield i+2

    def init_function_gen():
        yield 2 # start off on the even numbers then restart on odds
        while True:
            yield 3
    init_gen=init_function_gen()
    init_function=lambda: init_gen.next()
    
    
    num_evaluations,best_score,best=hillclimb.hillclimb_and_restart(init_function,move_operator,objective_function,max_evaluations)
    
    # should have used all iterations
    assert num_evaluations == max_evaluations
    # should have jumped onto odd numbers
    assert best == 101
    assert best_score == 101

def test_hillclimb_and_restart_getting_worse():
    '''
    see whether we restart on each number correctly
    '''
    def move_operator(i):
        # even numbers only go up to 50
        if i % 2 == 0 and i < 50:
            yield i+2
        elif i % 2 != 0 and i < 100: # odd numbers go higher
            yield i+2
    
    def init_function_gen():
        yield 3 # start off on the odd numbers then restart on evens
        while True:
            yield 2
    init_gen=init_function_gen()
    init_function=lambda: init_gen.next()
    
    num_evaluations,best_score,best=hillclimb.hillclimb_and_restart(init_function,move_operator,objective_function,max_evaluations)
    
    # should have used all iterations
    assert num_evaluations == max_evaluations
    # should have retained score from odd numbers
    assert best == 101
    assert best_score == 101
