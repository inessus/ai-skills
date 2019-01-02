import graphene

class Query(graphene.ObjectType):
    hello = graphene.String(argument=graphene.String(default_value="stranger"))

    def resolve_hello(self, info, argument):
        return 'this is Hello ' + argument

schema = graphene.Schema(query=Query)
result = schema.execute('{ hello }')
print(result.data['hello']) # "Hello stranger"

# or passing the argument in the query
result = schema.execute('{ hello (argument: "graph") }')
print(result.data['hello']) # "Hello graph"


class Episode(graphene.Enum):
    NEWHOPE = 4
    EMPIRE = 5
    JEDI = 6


Episode = graphene.Enum('Episode', [('NEWHOPE', 4), ('EMPIRE', 5), ('JEDI', 6)])
class Episode(graphene.Enum):
    NEWHOPE = 4
    EMPIRE = 5
    JEDI = 6

    @property
    def description(self):
        if self == Episode.NEWHOPE:
            return 'New Hope Episode'
        return 'Other episode'
schema = graphene.Enum.from_enum(Episode)

result = schema.execute('{ hello }')
print(result.data['hello']) # "Hello stranger"

if __name__ == "__main__":
    print("test")