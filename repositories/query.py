from hashlib import sha256, sha1

from pydantic import BaseModel, validator

from repositories.sampler import SamplerEnum


class Query(BaseModel):
    prompt: str
    seed: int = 42
    sampler: SamplerEnum = SamplerEnum.ddim
    iterations: int = 1
    scale: float = 7.5
    width: int = 512
    height: int = 512
    sampling_steps: int = 50

    def filename(self):
        data = self.json()
        return f"{sha1(data.encode('utf-8')).hexdigest()}.png"

    @validator('prompt')
    def name_must_not_be_empty(cls, v):
        if '' == v:
            raise ValueError('Prompt cannot be empty')
        return v

    @validator('width')
    def width_must_be_multiple_of_64(cls, v):
        if v % 64 != 0:
            raise ValueError('Width must be a multiple of 64')
        return v

    @validator('height')
    def height_must_be_multiple_of_64(cls, v):
        if v % 64 != 0:
            raise ValueError('Height must be a multiple of 64')
        return v
