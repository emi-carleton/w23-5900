import torch

from vmas.simulator.core import Agent, Landmark, Sphere, World, Box
from vmas.simulator.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        print("Airplane scenario")
        world = World(batch_dim=batch_dim, device=device, dim_c=3)
        # set any world properties first
        num_agents = 3 #two speakers, one listener
        num_landmarks = 2 #one starting, one destination city


        # Add airplane agent - listener
        speaker = True
        name = "Airplane" 
        agent = Agent(
            name=name,
            collide=True,
            movable=True,
            silent=True,
            shape=Sphere(radius=5),
        )
        world.add_agent(agent)


        # Add tower1 agent - speaker
        speaker = True
        name = "Tower1" 
        agent = Agent(
            name=name,
            collide=False,
            movable=False,
            silent=False,
            shape=Box(width=2,length=5),
        )
        world.add_agent(agent)
        
        # Add tower2 agent - speaker
        speaker = True
        name = "Tower2" 
        agent = Agent(
            name=name,
            collide=False,
            movable=False,
            silent=False,
            shape=Box(width=2,length=5),
        )
        world.add_agent(agent)

        
        # Add starting city landmark 
        landmark = Landmark(
            name="City1", collide=True, shape=Sphere(radius=1)
        )
        world.add_landmark(landmark)
        
        # Add destination city landmark
        landmark = Landmark(
            name="City2", collide=True, shape=Sphere(radius=1)
        )
        world.add_landmark(landmark)
        

        print(world.agents[0].action_size)

        return world

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            # assign goals to agents
            for agent in self.world.agents:
                agent.goal = None
            # want listener to go to the goal landmark - City 2
            self.world.agents[0].goal = self.world.landmarks[1]
           
            #setting colors for the plane
            self.world.agents[0].color = torch.tensor(
                    [0.10, 0.10, 0.10],
                    device=self.world.device,
                    dtype=torch.float32,
            )
            # setting colors for the towers
            self.world.agents[1].color = torch.tensor(
                    [0.25, 0.25, 0.25],
                    device=self.world.device,
                    dtype=torch.float32,
            )
            self.world.agents[2].color = torch.tensor(
                    [0.25, 0.25, 0.25],
                    device=self.world.device,
                    dtype=torch.float32,
            )
            # setting colors for the cities
            self.world.landmarks[0].color = torch.tensor(
                [0.45, 0.45, 0.45],
                device=self.world.device,
                dtype=torch.float32,
            )
            self.world.landmarks[1].color = torch.tensor(
                [0.45, 0.45, 0.45],
                device=self.world.device,
                dtype=torch.float32,
            )


        #Set Airplane starting position to (0,0)
        self.world.agents[0].set_pos(
            torch.zeros(
                (
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p)
                ),
                device=self.world.device,
                dtype=torch.float32,
            ),
            batch_index=env_index,
        )
        print("Airplane stating position: ", self.world.agents[0].state.pos)

        #Set Tower1 position to (10,0)
        if env_index is not None:
            self.world.agents[1].set_pos(
                torch.tensor([[10.0,0.0]],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        else:
            x = torch.tensor([[10.0,0.0]],device=self.world.device, dtype=torch.float32,)
            repeatX = x.repeat(self.world.batch_dim,1)
            self.world.agents[1].set_pos(
                repeatX,
                batch_index=env_index,
            )
        print("Tower 1 position: ", self.world.agents[1].state.pos)

        #Set Tower2 position to (20,0)
        if env_index is not None:
            self.world.agents[2].set_pos(
                torch.tensor([[20.0,0.0]],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        else:
            x = torch.tensor([[20.0,0.0]],device=self.world.device, dtype=torch.float32,)
            repeatX = x.repeat(self.world.batch_dim,1)
            self.world.agents[2].set_pos(
                repeatX,
                batch_index=env_index,
            )
        print("Tower 2 position: ", self.world.agents[2].state.pos)
            
        #Set City1 position (0,0)
        self.world.landmarks[0].set_pos(
            torch.zeros(
                (
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p)
                ),
                device=self.world.device,
                dtype=torch.float32,
            ),
            batch_index=env_index,
        )
        print("City 1 position: ", self.world.landmarks[0].state.pos)

        #Set City 2 position (30,0)
        if env_index is not None:
            self.world.landmarks[1].set_pos(
                torch.tensor([[30.0,0.0]],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        else:
            x = torch.tensor([[30.0,0.0]],device=self.world.device, dtype=torch.float32,)
            repeatX = x.repeat(self.world.batch_dim,1)
            self.world.landmarks[1].set_pos(
                repeatX,
                batch_index=env_index,
            )
        print("City 2 position: ", self.world.landmarks[1].state.pos)


    def observation(self, agent):
        print("in observation")
        # speaker's observation - the distance between the tower and the plane
        if not agent.movable:
            print("in speaker observation")
            if(agent.name == "Tower1"):
                tower_plane_distance = torch.sqrt(
                    torch.sum(torch.square(self.world.agents[0].state.pos-self.world.agents[1].state.pos),dim=-1))
                #When the airplane is out of the tower's range, set the distance to an arbitrarily big value
                if(self.world.agents[0].state.pos >15):
                    tower_plane_distance = torch.tensor([10000],device=self.world.device, dtype=torch.float32,)
                
            elif(agent.name == "Tower2"):
                tower_plane_distance = torch.sqrt(
                    torch.sum(torch.square(self.world.agents[0].state.pos-self.world.agents[1].state.pos),dim=-1))
                 #When the airplane is out of the tower's range, set the distance to an arbitrarily big value
                if(self.world.agents[0].state.pos < 15):
                    tower_plane_distance = torch.tensor([10000],device=self.world.device, dtype=torch.float32,)
            plane_pos = self.world.agents[0].state.pos

            print("tower_plane_distance", tower_plane_distance)
            return torch.cat((tower_plane_distance,plane_pos), dim=-1)
        
        # listener
        if agent.silent:
            print("in listener observation")
            # communication of all other agents
            comm = []
            for other in self.world.agents:
                if other is agent or (other.state.c is None):
                    continue
                comm.append(other.state.c)
            #airplane's position difference with city2
            distance_to_city2 = torch.subtract(self.world.agents[0].state.pos, self.world.landmarks[1].state.pos)

            return torch.cat(([*comm],distance_to_city2,self.world.agents[0].torque,self.world.agents[0].velocity),dim=-1)
        

    def reward(self, agent: Agent):
        self.rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        # sqaured distance between airplane and city2
        self.rew += torch.sqrt(
                    torch.sum(torch.square(self.world.agents[0].state.pos-self.world.landmarks[1].state.pos),dim=-1))
        # reward for staying in the arc of towers 1 
        if(self.world.agents[0].state.pos < 15):
            self.rew += torch.sqrt(
                        torch.sum(torch.square(self.world.agents[0].state.pos-self.world.agents[1].state.pos),dim=-1))
        # reward for staying in the arc of towers 2 
        else:
            self.rew += torch.sqrt(
                        torch.sum(torch.square(self.world.agents[0].state.pos-self.world.agents[2].state.pos),dim=-1))
        
        print("reward: ", self.rew)
        return self.rew    