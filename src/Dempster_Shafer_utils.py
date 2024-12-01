__author__='Lucas DEREGNAUCOURT'
__author_email__='lucas.deregnaucourt@insa-rouen.fr'

import torch

class Dempster_layer(torch.nn.Module):
    '''
    Apply the scalable Dempster fusion
    '''
    def __init__(self):
        super(Dempster_layer, self).__init__()
   
    def forward(self,m):
        m_omega = torch.cat((m[:,:-1,:,...]+m[:,None,-1,:,...],m[:,None,-1,:,...]),1)
        tmp = torch.sum(torch.log(torch.relu(m_omega)+1e-10),2)
        tmp = tmp-tmp.max(1,keepdim=True).values
        
        m1 = torch.exp(tmp)
        prodOmega = m1[:,-1,None]
        m1 = m1[:,:-1, ...] - prodOmega
        
        m1 = torch.cat((m1,prodOmega),1) 
        m1 = torch.nn.functional.normalize(m1,p=1.,dim=1)

        return m1

class DSmP(torch.nn.Module):
    '''
    Apply the Dezert-Smarandache probability transformation
    '''
    def __init__(self, eps):
        super(DSmP, self).__init__()
        self.eps = eps
   
    def forward(self, x):
        return (x+(x*x[:,None,-1, ...]+self.eps*x[:,None,-1, ...])/(1-x[:,None,-1, ...]+self.eps*(x.shape[1]-1)))[:,:-1]
     

class Discounting_layer(torch.nn.Module):
    '''
    Apply the adaptive discounting procedure:
    1. Compute the pairwise distances
    2. Compute the pairwise conflicts
    3. Compute the discounting coefficients
    4. Discount the mass functions
    '''
    def __init__(self,nb_classe,nb_expert):
        super(Discounting_layer,self).__init__()

        self.nb_classe = nb_classe
        self.nb_expert = nb_expert

        '''
        Indices matrix for the pairwise distances computation
        '''
        count = 0
        indices = torch.zeros((nb_expert, nb_expert-1), dtype=int)
        for i in range(nb_expert-1):
            for j in range(i+1, nb_expert):
                    indices[i, j-1] = count
                    indices[j, i] = count
                    count+=1
        
        tab_indices = torch.triu_indices(self.nb_expert,self.nb_expert,offset=1)
        self.const = 1 - (2*self.nb_classe+1)/(self.nb_classe+1)**2 #Constant used to compute the conflicts from the distances
        self.lamb = 2 #lambda parameter for the discounting coefficients computation

        self.register_buffer("tab_indices", tab_indices)
        self.register_buffer("indices", indices)


    def forward(self,x, output_conflict=False, output_discouting_coefs=False):      

        '''
        Jousselme's distance
        '''
        m = torch.index_select(x,2,self.tab_indices[0]) - torch.index_select(x,2,self.tab_indices[1])
        m = torch.sum(m**2,1) + 2./self.nb_classe * m[:,-1,...]*torch.sum(m[:,:-1,...],1)

        '''
        Conflicts derived from the distances
        '''
        conf = self.const * torch.sqrt(m/2.)

        '''
        Discouting coefficients derived from the conflicts
        '''
        conf_per_modality = conf[:, self.indices, ...].sum(dim=2)
        conf_per_modality = (conf_per_modality / (conf_per_modality.shape[1]-1))
        discountings = ((1-conf_per_modality.unsqueeze(1)**self.lamb)**(1/self.lamb))

        '''
        Mass functions discounting
        '''
        new_x_om = x[:, :-1, :, ...] * discountings
        new_x = torch.clamp(torch.cat((new_x_om, 1-new_x_om.sum(dim=1).unsqueeze(dim=1)), dim=1), min=0., max=1.)

        return new_x, conf_per_modality if output_conflict else None, discountings.squeeze(1) if output_discouting_coefs else None

